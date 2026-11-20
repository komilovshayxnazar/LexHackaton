#!/usr/bin/env python3
"""Build and query the LexUZ FAISS + BM25 hybrid index."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
OUT = ROOT / "vector_index"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOKEN_RE = re.compile(r"[\w]+(?:[ʻʼ'’‘-][\w]+)*", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ARTICLE_QUERY_RE = re.compile(
    r"\b(\d+(?:\s*-\s*\d+|\s+\d+)?)\s*[- ]?\s*modda(?:sida|si|ning|ga|dan|ni|lar)?\b",
    re.I,
)
RRF_K = 60
# Exact "N-modda" matches are pinned above fused scores. Normative article
# chunks outrank amendment notes that merely mention the same article.
EXACT_ARTICLE_BOOST = {"article": 1.0, "other": 0.5}


def load_model():
    # Imported lazily: torch/sentence-transformers dominate start-up time and
    # are not needed for input validation or argument parsing.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_inputs(law, articles_file, tables_file, amendments_file):
    errors = []
    warnings = []
    required = {"metadata", "chapters", "tables", "annexes", "amendments"}
    if not isinstance(law, dict) or not required.issubset(law):
        errors.append("law_structured.json is missing required top-level fields")
        return errors, warnings, []

    articles = []
    ids = set()
    article_numbers = set()
    for chapter in law.get("chapters", []):
        for article in chapter.get("articles", []):
            articles.append(article)
            article_id = article.get("article_id")
            number = article.get("article_number")
            if not article_id:
                errors.append(f"Article missing article_id: {number or '<unknown>'}")
            elif article_id in ids:
                errors.append(f"Duplicate article_id: {article_id}")
            else:
                ids.add(article_id)
            if not number:
                errors.append(f"Article missing article_number: {article_id or '<unknown>'}")
            elif number in article_numbers:
                warnings.append(f"Repeated article_number: {number}")
            else:
                article_numbers.add(number)
            if not str(article.get("text") or "").strip():
                warnings.append(f"Empty article text: {article_id or number or '<unknown>'}")
            part_numbers = [
                str(part.get("part_number"))
                for part in article.get("parts", [])
                if part.get("part_number") is not None
            ]
            duplicates = sorted(
                number for number, count in Counter(part_numbers).items() if count > 1
            )
            if duplicates:
                warnings.append(
                    f"Repeated part numbers in {number or article_id}: {', '.join(duplicates)}"
                )

    for table in law.get("tables", []):
        if not table.get("columns"):
            warnings.append(f"Table has no column headings: {table.get('table_id', '<unknown>')}")
        if not table.get("rows"):
            warnings.append(f"Table has no rows: {table.get('table_id', '<unknown>')}")
        for row in table.get("rows", []):
            if row.get("row_type") == "rate" and not row.get("item") and not row.get("rate"):
                warnings.append(
                    f"Rate row has neither item nor rate: "
                    f"{table.get('table_id', '<unknown>')} row {row.get('row_index')}"
                )
    if not articles:
        errors.append("No articles found in law_structured.json")

    for label, path, value in (
        ("articles.json", PROCESSED / "articles.json", articles_file),
        ("tables.json", PROCESSED / "tables.json", tables_file),
        ("amendments.json", PROCESSED / "amendments.json", amendments_file),
    ):
        if path.exists() and value is None:
            errors.append(f"{label} exists but could not be loaded")
        elif not path.exists():
            warnings.append(f"Optional input is missing: {label}")
    return errors, warnings, articles


_TOKEN_COUNT_CACHE: dict[tuple[int, str], int] = {}


def tokenizer_count(tokenizer, text: str) -> int:
    """Token count incl. special tokens, memoised (chunking re-counts the same strings)."""
    key = (id(tokenizer), text)
    count = _TOKEN_COUNT_CACHE.get(key)
    if count is None:
        count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        _TOKEN_COUNT_CACHE[key] = count
    return count


def split_oversized_unit(text, prefix, tokenizer, max_tokens):
    """Split only exceptionally long legal sentences, retaining all source words."""
    if tokenizer_count(tokenizer, f"passage: {prefix} {text}") <= max_tokens:
        return [text]

    clauses = re.split(r"(?<=[,;:])\s+", text)
    result = []
    current = ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip()
        if current and tokenizer_count(tokenizer, f"passage: {prefix} {candidate}") > max_tokens:
            result.append(current)
            current = clause
        else:
            current = candidate
        if tokenizer_count(tokenizer, f"passage: {prefix} {current}") > max_tokens:
            words = current.split()
            current = ""
            word_chunk = ""
            for word in words:
                candidate = f"{word_chunk} {word}".strip()
                if word_chunk and tokenizer_count(
                    tokenizer, f"passage: {prefix} {candidate}"
                ) > max_tokens:
                    result.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = candidate
            current = word_chunk
    if current:
        result.append(current)
    return result


def chunk_text(text, prefix, tokenizer, max_tokens, target_tokens, overlap_tokens):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    units = []
    for sentence in sentences:
        units.extend(split_oversized_unit(sentence, prefix, tokenizer, max_tokens))

    chunks = []
    current = []
    for unit in units:
        candidate = " ".join(current + [unit])
        if current and (
            tokenizer_count(tokenizer, f"passage: {prefix} {candidate}") > target_tokens
        ):
            chunks.append(" ".join(current))
            previous = current[:]
            overlap = []
            while previous:
                trial = " ".join([previous[-1], *overlap])
                trial_count = tokenizer_count(tokenizer, f"passage: {prefix} {trial}")
                if trial_count > overlap_tokens:
                    break
                overlap = [previous.pop(), *overlap]
            current = overlap
            candidate = " ".join(current + [unit])
            while current and tokenizer_count(
                tokenizer, f"passage: {prefix} {candidate}"
            ) > max_tokens:
                current.pop(0)
                candidate = " ".join(current + [unit])
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def tokenize(text: str):
    return [word.lower() for word in TOKEN_RE.findall(text)]


@lru_cache(maxsize=None)
def article_number_key(value):
    value = str(value or "").lower().replace("–", "-")
    value = re.sub(r"modda(?:sida|si|ning|ga|dan|ni|lar)?", "", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", "-", value)
    return value.strip("-")


def build_chunks(law, articles, tokenizer, max_length):
    metadata = law["metadata"]
    law_title = metadata.get("law_title") or ""
    law_number = metadata.get("law_number") or ""
    max_tokens = min(480, max_length)
    target_tokens = min(420, max_tokens - 16)
    overlap_tokens = min(75, max(0, target_tokens // 6))
    chunks = []

    def add_source(text, base, chunk_type, article=None, part=None, reference=None, heading=None):
        chapter = base.get("chapter")
        article_number = article.get("article_number") if article else base.get("article")
        title = article.get("article_title") if article else None
        prefix_fields = [law_title, law_number, chapter, article_number, title, heading, part]
        prefix = " ".join(str(x) for x in prefix_fields if x)
        pieces = chunk_text(
            text, prefix, tokenizer, max_tokens, target_tokens, overlap_tokens
        )
        for piece in pieces:
            chunks.append(
                {
                    "_body": piece,
                    "_prefix": prefix,
                    "law_title": law_title,
                    "law_number": law_number,
                    "chapter": chapter,
                    "article": article_number,
                    "part": part,
                    "paragraph": None,
                    "chunk_type": chunk_type,
                    "language": metadata.get("language") or "uz",
                    "source": metadata.get("source") or "LexUZ",
                    "source_reference": reference,
                    "document_version": None,
                    "effective_from": None,
                    "article_title": title,
                }
            )

    chapter_titles = {
        chapter.get("chapter_id"): chapter.get("chapter_title")
        for chapter in law.get("chapters", [])
    }
    for article in articles:
        article_number = article.get("article_number")
        source_ref = f"{law_number}, {article_number}" if article_number else law_number
        chapter_number = article.get("chapter_number")
        chapter_title = chapter_titles.get(article.get("chapter_id"))
        chapter = f"{chapter_number}-bob" if chapter_number else None
        if chapter_title:
            chapter = f"{chapter} — {chapter_title}" if chapter else chapter_title
        base = {"chapter": chapter, "article": article_number}
        parts = article.get("parts") or []
        if parts:
            for index, part_obj in enumerate(parts, 1):
                part_text = part_obj.get("text") or part_obj.get("lead_text") or ""
                part_value = part_obj.get("part_number")
                part_label = str(part_value) if part_value is not None else str(index)
                add_source(
                    part_text,
                    base,
                    "article",
                    article=article,
                    part=part_label,
                    reference=f"{source_ref}, {part_label}-qism",
                )
        else:
            add_source(
                article.get("text") or "",
                base,
                "article",
                article=article,
                reference=source_ref,
            )

    for table_index, table in enumerate(law.get("tables", []), 1):
        table_title = table.get("title") or f"Jadval {table_index}"
        rows = table.get("rows", [])
        # Parent rows ("a) ... boʻyicha") carry no rate; their wording is folded
        # into each child rate row below. Only parents without any rate row
        # are indexed on their own, so no source text is lost.
        parents_with_rates = {
            (row.get("section"), row.get("parent"))
            for row in rows
            if row.get("row_type") == "rate" and row.get("parent")
        }
        for row_index, row in enumerate(rows, 1):
            row_type = row.get("row_type")
            section = row.get("section")
            item = row.get("item")
            rate = row.get("rate")
            note = row.get("note")
            parent = row.get("parent")
            if row_type == "section" and not rate:
                continue  # heading only; repeated in every row of the section
            if row_type == "parent" and (section, parent) in parents_with_rates:
                continue
            if row_type == "section":
                item = None  # numbered section that carries its own rate
            band = " — ".join(x for x in (parent, item) if x)
            text_parts = [
                f"Boʻlim: {section}" if section else None,
                f"Band: {band}" if band else None,
                f"Stavka: {rate}" if rate else None,
                f"Izoh: {note}" if note else None,
            ]
            row_text = ". ".join(p for p in text_parts if p)
            if not row_text:
                row_text = " | ".join(
                    str(cell).strip() for cell in row.get("cells") or [] if str(cell).strip()
                )
            if not row_text:
                continue
            base = {"chapter": None, "article": None}
            add_source(
                row_text,
                base,
                "table",
                reference=f"{law_number}, ilova: {table_title}, qator {row.get('row_index', row_index - 1)}",
                heading=f"Jadval: {table_title}",
            )

    for amendment_index, amendment in enumerate(law.get("amendments", []), 1):
        body = amendment.get("content") or ""
        if not body.strip():
            continue
        article_number = amendment.get("context_article")
        base = {"chapter": None, "article": article_number}
        add_source(
            body,
            base,
            "other",
            reference=f"{law_number}, o‘zgartirish {amendment.get('amendment_id', amendment_index)}",
        )

    counters = Counter()
    for chunk in chunks:
        if chunk["chunk_type"] == "article":
            key = re.sub(r"\W+", "_", article_number_key(chunk["article"])).strip("_")
            prefix = f"ORQ600_ART_{key}"
        elif chunk["chunk_type"] == "table":
            prefix = "ORQ600_TABLE"
        else:
            prefix = "ORQ600_AMD"
        counters[prefix] += 1
        chunk["chunk_id"] = f"{prefix}_CHUNK_{counters[prefix]:04d}"
        chunk["text"] = chunk.pop("_body")
        chunk["_embedding_text"] = (
            "passage: " + " ".join(v for v in (chunk.pop("_prefix"), chunk["text"]) if v)
        )
        chunk["token_count"] = tokenizer_count(tokenizer, chunk["_embedding_text"])
    return chunks, {
        "target_chunk_tokens": target_tokens,
        "maximum_chunk_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
    }


def rrf_search(query, chunks, index, bm25, model, top_k=10):
    query_vector = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    vector_count = min(max(top_k * 10, 50), len(chunks))
    scores, semantic_ids = index.search(query_vector, vector_count)
    semantic_ranks = {
        int(item): rank
        for rank, item in enumerate(semantic_ids[0], 1)
        if item >= 0
    }
    lexical_scores = bm25.get_scores(tokenize(query))
    top = np.argpartition(-lexical_scores, vector_count - 1)[:vector_count]
    lexical_order = top[np.argsort(-lexical_scores[top], kind="stable")]
    lexical_ranks = {int(item): rank for rank, item in enumerate(lexical_order, 1)}

    exact_article = ARTICLE_QUERY_RE.search(query)
    article_key = article_number_key(exact_article.group(1)) if exact_article else None
    combined = set(semantic_ranks) | set(lexical_ranks)
    if article_key:
        # An explicitly named article must be retrievable even if neither
        # ranker put it in its candidate window.
        combined.update(
            idx for idx, chunk in enumerate(chunks)
            if article_number_key(chunk.get("article")) == article_key
        )
    ranked = []
    for chunk_idx in combined:
        rrf_score = (
            (1 / (RRF_K + semantic_ranks[chunk_idx]) if chunk_idx in semantic_ranks else 0)
            + (1 / (RRF_K + lexical_ranks[chunk_idx]) if chunk_idx in lexical_ranks else 0)
        )
        chunk = chunks[chunk_idx]
        exact_match = bool(
            article_key and article_number_key(chunk.get("article")) == article_key
        )
        if exact_match:
            rrf_score += EXACT_ARTICLE_BOOST.get(chunk.get("chunk_type"), 0.5)
        ranked.append(
            {
                "chunk_index": chunk_idx,
                "rrf_score": rrf_score,
                "semantic_score": float(scores[0][semantic_ranks[chunk_idx] - 1])
                if chunk_idx in semantic_ranks
                else None,
                "bm25_score": float(lexical_scores[chunk_idx]),
                "semantic_rank": semantic_ranks.get(chunk_idx),
                "bm25_rank": lexical_ranks.get(chunk_idx),
                "exact_article_match": exact_match,
            }
        )
    ranked.sort(key=lambda result: result["rrf_score"], reverse=True)
    results = []
    for result in ranked[:top_k]:
        chunk = chunks[result["chunk_index"]]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "article": chunk["article"],
                "chunk_type": chunk["chunk_type"],
                "text": chunk["text"],
                "similarity_score": result["semantic_score"],
                "bm25_score": result["bm25_score"],
                "bm25_rank": result["bm25_rank"],
                "final_rrf_score": result["rrf_score"],
                "source_reference": chunk["source_reference"],
                "exact_article_match": result["exact_article_match"],
            }
        )
    return results


def evaluation_cases(chunks):
    def find_chunk(chunk_type, article=None, text_contains=None):
        for chunk in chunks:
            if chunk["chunk_type"] != chunk_type:
                continue
            if article and chunk["article"] != article:
                continue
            if text_contains and text_contains.lower() not in chunk["text"].lower():
                continue
            return chunk["chunk_id"]
        return None

    cases = [
        {
            "query": "1-modda ushbu Qonunning maqsadi nima?",
            "relevant_chunk_ids": [find_chunk("article", "1-modda")],
            "expected_reference": "OʻRQ-600, 1-modda",
        },
        {
            "query": "Yuridik ahamiyatga molik harakatlar uchun undiriladigan majburiy toʻlov nima?",
            "relevant_chunk_ids": [find_chunk("article", "3-modda")],
            "expected_reference": "OʻRQ-600, 3-modda",
        },
        {
            "query": "Fuqarolik sudiga mulkiy daʼvo arizasi berilganda davlat boji stavkasi qancha?",
            "relevant_chunk_ids": [
                find_chunk("table", text_contains="daʼvo bahosining 4 foizi")
            ],
            "expected_reference": "OʻRQ-600, ilova: fuqarolik sudlari boji stavkasi",
        },
        {
            "query": "OʻRQ-600 qonunining 5-moddasida qaysi davlat boji stavkalari belgilangan?",
            "relevant_chunk_ids": [
                chunk["chunk_id"]
                for chunk in chunks
                if chunk["chunk_type"] == "article" and chunk["article"] == "5-modda"
            ],
            "expected_reference": "OʻRQ-600, 5-modda",
        },
    ]
    for case in cases:
        case["relevant_chunk_ids"] = [
            item for item in case["relevant_chunk_ids"] if item
        ]
    return [case for case in cases if case["relevant_chunk_ids"]]


def metric_report(cases, results_by_query, ks=(5, 10)):
    metrics = {}
    for k in ks:
        precision = recall = mrr = ndcg = hits = 0.0
        for case in cases:
            relevant = set(case["relevant_chunk_ids"])
            retrieved = [
                item["chunk_id"] for item in results_by_query[case["query"]][:k]
            ]
            relevant_ranks = [
                index + 1 for index, chunk_id in enumerate(retrieved) if chunk_id in relevant
            ]
            count = len(relevant_ranks)
            precision += count / k
            recall += count / len(relevant)
            mrr += 1 / min(relevant_ranks) if relevant_ranks else 0
            dcg = sum(1 / np.log2(rank + 1) for rank in relevant_ranks)
            ideal = sum(
                1 / np.log2(rank + 1)
                for rank in range(1, min(len(relevant), k) + 1)
            )
            ndcg += dcg / ideal if ideal else 0
            hits += float(bool(relevant_ranks))
        total = max(1, len(cases))
        metrics[str(k)] = {
            f"precision_at_{k}": precision / total,
            f"recall_at_{k}": recall / total,
            "mrr": mrr / total,
            f"ndcg_at_{k}": ndcg / total,
            f"hit_rate_at_{k}": hits / total,
        }
    return metrics


def atomic_write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def atomic_save_numpy(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
        temp_path = Path(tmp.name)
    np.save(temp_path, array)
    temp_path.replace(path)


def build_index():
    REPORTS.mkdir(parents=True, exist_ok=True)
    validation_report_path = REPORTS / "indexing_validation_report.json"
    loaded = {}
    errors = []
    warnings = []
    for name in ("law_structured.json", "articles.json", "tables.json", "amendments.json"):
        path = PROCESSED / name
        if not path.exists():
            if name == "law_structured.json":
                errors.append(f"Required input is missing: {path}")
            else:
                warnings.append(f"Optional input is missing: {name}")
            continue
        try:
            loaded[name] = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"Invalid JSON in {name}: {exc}")

    law = loaded.get("law_structured.json")
    if law is not None:
        input_errors, input_warnings, articles = validate_inputs(
            law,
            loaded.get("articles.json"),
            loaded.get("tables.json"),
            loaded.get("amendments.json"),
        )
        errors.extend(input_errors)
        warnings.extend(input_warnings)
    else:
        articles = []

    validation_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"exists": (PROCESSED / name).exists(), "valid_json": name in loaded}
            for name in (
                "law_structured.json",
                "articles.json",
                "tables.json",
                "amendments.json",
            )
        },
        "article_count": len(articles),
        "errors": errors,
        "warnings": warnings,
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
    }
    write_json(validation_report_path, validation_report)
    if errors:
        raise RuntimeError(
            f"Input validation failed; see {validation_report_path}"
        )

    model = load_model()
    tokenizer = model.tokenizer
    max_length = int(model.max_seq_length)
    chunks, chunking = build_chunks(law, articles, tokenizer, max_length)
    if not chunks:
        raise RuntimeError("No non-empty legal chunks were generated")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Generated duplicate chunk IDs")

    passages = [chunk.pop("_embedding_text") for chunk in chunks]
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
        convert_to_numpy=True,
    ).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape[0] != len(chunks) or not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("Embedding count or normalization validation failed")
    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    if index.ntotal != len(chunks):
        raise RuntimeError("FAISS vector count does not match chunk metadata")

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    mapping = {
        chunk["chunk_id"]: idx for idx, chunk in enumerate(chunks)
    }
    cases = evaluation_cases(chunks)
    evaluated = {}
    for case in cases:
        evaluated[case["query"]] = rrf_search(
            case["query"], chunks, index, bm25, model, top_k=10
        )
    evaluations = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_case_count": len(cases),
        "metrics": metric_report(cases, evaluated),
        "cases": [
            {
                **case,
                "results": evaluated[case["query"]],
                "hit_at_5": bool(
                    set(case["relevant_chunk_ids"])
                    & {item["chunk_id"] for item in evaluated[case["query"]][:5]}
                ),
                "hit_at_10": bool(
                    set(case["relevant_chunk_ids"])
                    & {item["chunk_id"] for item in evaluated[case["query"]][:10]}
                ),
            }
            for case in cases
        ],
    }

    now = datetime.now(timezone.utc).isoformat()
    model_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    config = {
        "embedding_model": MODEL_NAME,
        "model_revision": model_revision,
        "embedding_dimension": dimension,
        "vector_index_type": "IndexFlatIP",
        "similarity_metric": "cosine_similarity (normalized inner product)",
        "chunking_strategy": "structure-aware; articles, parts, table rows, and amendments",
        "tokenizer_max_sequence_length": max_length,
        **chunking,
        "overlap_policy": "overlap complete sentence/clause units when within token budget",
        "dataset_version": {
            name: hashlib.sha256(
                (PROCESSED / name).read_bytes()
            ).hexdigest()
            for name in (
                "law_structured.json",
                "articles.json",
                "tables.json",
                "amendments.json",
            )
            if (PROCESSED / name).exists()
        },
        "created_at": now,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("sentence-transformers", "faiss-cpu", "rank-bm25", "numpy")
        },
    }
    index_config = {
        "metadata": chunks,
        "mapping": mapping,
        "index": index,
        "embeddings": embeddings,
        "bm25": bm25,
        "tokenized_corpus": tokenized_corpus,
        "config": config,
    }
    if len(index_config["metadata"]) != len(index_config["mapping"]):
        raise RuntimeError("Metadata-to-chunk mapping validation failed")

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_save_numpy(OUT / "embeddings" / "law_embeddings.npy", embeddings)
    faiss_path = OUT / "faiss" / "law_index.faiss"
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=faiss_path.parent, delete=False) as tmp:
        temp_index_path = Path(tmp.name)
    faiss.write_index(index, str(temp_index_path))
    temp_index_path.replace(faiss_path)
    write_json(OUT / "metadata" / "chunks.json", chunks)
    write_json(OUT / "metadata" / "chunk_mapping.json", mapping)
    write_json(OUT / "config" / "index_config.json", config)
    write_json(OUT / "reports" / "retrieval_evaluation.json", evaluations)
    write_json(validation_report_path, validation_report)

    # Kept for compatibility with downstream consumers; query mode rebuilds
    # BM25 from chunks.json instead of unpickling it.
    bm25_path = OUT / "bm25" / "bm25_index.pkl"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        bm25_path,
        pickle.dumps(
            {
                "bm25": bm25,
                "tokenized_corpus": tokenized_corpus,
                "chunk_ids": chunk_ids,
                "tokenizer": "unicode word tokens; preserves Uzbek characters and numbers",
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
    )

    chars = [len(chunk["text"]) for chunk in chunks]
    token_counts = [chunk["token_count"] for chunk in chunks]
    indexing_report = {
        "generated_at": now,
        "status": "completed",
        "input_documents": 1,
        "inputs_used": [
            str(PROCESSED / name)
            for name in (
                "law_structured.json",
                "articles.json",
                "tables.json",
                "amendments.json",
            )
            if (PROCESSED / name).exists()
        ],
        "article_count": len(articles),
        "table_count": len(law.get("tables", [])),
        "amendment_count": len(law.get("amendments", [])),
        "chunk_count": len(chunks),
        "chunk_counts_by_type": dict(Counter(c["chunk_type"] for c in chunks)),
        "average_chunk_characters": round(sum(chars) / len(chars), 2),
        "average_chunk_tokens_including_passage_prefix": round(
            sum(token_counts) / len(token_counts), 2
        ),
        "oversized_chunks": [
            chunk["chunk_id"]
            for chunk in chunks
            if chunk["token_count"] > max_length
        ],
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dimension,
        "faiss_index": {"type": "IndexFlatIP", "vectors": int(index.ntotal)},
        "bm25_index": {"status": "built", "documents": len(tokenized_corpus)},
        "retrieval_evaluation": evaluations["metrics"],
        "errors": [],
        "warnings": validation_report["warnings"],
        "validation_report": str(validation_report_path.relative_to(ROOT)),
    }
    write_json(OUT / "reports" / "indexing_report.json", indexing_report)
    print(json.dumps(indexing_report, ensure_ascii=False, indent=2))


def load_bm25(chunks):
    """Rebuild BM25 from chunk text: milliseconds for this corpus, and it avoids
    unpickling a file (pickle.load can execute arbitrary code)."""
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def query_index(query: str, top_k: int):
    chunks = read_json(OUT / "metadata" / "chunks.json")
    index = faiss.read_index(str(OUT / "faiss" / "law_index.faiss"))
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index/metadata mismatch: {index.ntotal} vectors vs {len(chunks)} chunks; rebuild the index"
        )
    bm25 = load_bm25(chunks)
    model = load_model()
    results = rrf_search(query, chunks, index, bm25, model, top_k=top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run hybrid retrieval instead of rebuilding")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        if args.query:
            query_index(args.query, args.top_k)
        else:
            build_index()
    except Exception as exc:
        print(f"vector indexing failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()