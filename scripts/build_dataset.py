#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexUZ HTML (.doc) -> structured JSON -> Gemma 4 SFT dataset.

Pipeline (docs/workflow.md):
  1. Document reading / format detection
  2. Structure detection (chapters, articles, amendments, annex table)
  3. Text + table extraction
  4. Cleaning / whitespace normalisation
  5. law_structured.json + articles.json + tables.json + amendments.json
  6. SFT dataset generation (grounded, source-quoted)
  7. Grouped train/validation/test split (no leakage)
  8. Validation
  9. Reports
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, OrderedDict
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source_document" / "O‘RQ-600 06.01.2020.doc"

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DATASET = DATA / "dataset"
REPORTS = DATA / "reports"

for d in (RAW, PROCESSED, DATASET, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONTHS = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6,
    "iyul": 7, "avgust": 8, "sentabr": 9, "sentyabr": 9, "oktabr": 10,
    "noyabr": 11, "dekabr": 12,
}

WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    """Collapse whitespace only; never alter characters or wording."""
    return WS_RE.sub(" ", s or "").strip()


UZ_LETTERS = r"A-Za-zʻʼ‘’"
UZ_DATE_RE = re.compile(rf"(\d{{4}})-yil\s+(\d{{1,2}})-([{UZ_LETTERS}]+)")


def month_num(mon: str) -> int:
    """Month number from an Uzbek month word with case suffix ('iyuldagi' -> 7); 0 if unknown."""
    mon = (mon or "").lower()
    for name, num in MONTHS.items():
        if mon.startswith(name):
            return num
    return 0


def iso_date(year, day, mon):
    num = month_num(mon)
    return f"{int(year):04d}-{num:02d}-{int(day):02d}" if num else None


def uz_date(text: str):
    """'2026-yil 13-iyuldagi' -> '2026-07-13' (best effort, else None)."""
    m = UZ_DATE_RE.search(text or "")
    return iso_date(m.group(1), m.group(2), m.group(3)) if m else None


def table_group(section: str) -> str:
    """Stable split group for one annex-table section."""
    return "TBL_" + hashlib.sha1(section.encode("utf-8")).hexdigest()[:8]


def norm_article_num(raw: str) -> str:
    """'22 1 ' -> '22-1'; '5' -> '5'."""
    raw = norm(raw)
    return re.sub(r"\s+", "-", raw)


def article_id(num: str) -> str:
    if "-" in num:
        base, sub = num.split("-", 1)
        return f"ORQ600_ART_{int(base):03d}_{sub}"
    return f"ORQ600_ART_{int(num):03d}"


def text_of(el) -> str:
    return norm(el.get_text(" ", strip=True))


# ---------------------------------------------------------------------------
# 1-4. Read + extract
# ---------------------------------------------------------------------------

html = SOURCE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "lxml")
body = soup.find("body")
container = body.find("div", id="divCont")
kids = container.find_all(recursive=False)
# Each element's classes and normalised text are needed by several passes
# below; compute them once instead of re-walking the DOM every time.
KID_INFO = [(el, el.get("class") or [], text_of(el)) for el in kids]

SOURCE_ARTICLE_HEADINGS = [t for _, classes, t in KID_INFO if "CLAUSE_DEFAULT" in classes]

# Full plain-text view of the document, used to verify every source quote
# against the original file (independent of the structured extraction).
FULL_TEXT = norm(body.get_text(" ", strip=True))

# --- document metadata ------------------------------------------------------
meta = {
    "law_title": None,
    "law_number": None,
    "adoption_date": None,
    "language": "uz",
    "source": "LexUZ",
    "document_type": "law",
}
form_labels = []
parliament_raw = None
signature = None
place = None
date_raw = None
number_raw = None
publication_origin = None

for el, classes, t in KID_INFO:
    if not t:
        continue
    if "ACT_FORM" in classes:
        form_labels.append(t)
    elif "ACT_TITLE" in classes and meta["law_title"] is None:
        meta["law_title"] = t
    elif "GRIF_PARLAMENT" in classes:
        parliament_raw = t
    elif "SIGNATURE" in classes:
        signature = t
    elif "ACT_ESSENTIAL_ELEMENTS" in classes:
        if re.search(r"\d{4}-yil", t):
            date_raw = t
        else:
            place = t
    elif "ACT_ESSENTIAL_ELEMENTS_NUM" in classes:
        number_raw = t
    elif "PUBLICATION_ORIGIN" in classes:
        publication_origin = t
    if "APPL_BANNER_LANDSCAPE_TITLE" in classes:
        break

if form_labels:
    meta["document_form"] = form_labels[0]
if parliament_raw:
    dates = UZ_DATE_RE.findall(parliament_raw)
    if dates:
        meta["parliament_adopted_date"] = iso_date(*dates[0])
    if len(dates) > 1:
        meta["senate_approved_date"] = iso_date(*dates[1])
if signature:
    meta["signatory"] = signature
if place:
    meta["place"] = place.rstrip(", ")
if date_raw:
    meta["adoption_date"] = uz_date(date_raw)
if number_raw:
    m = re.search(r"([A-Za-zʻʼ\u2018\u2019]{1,10}-\d+)-son", number_raw)
    if m:
        meta["law_number"] = m.group(1)
if publication_origin:
    meta["publication_origin"] = publication_origin
meta["source_file"] = SOURCE.name
meta["detected_format"] = "HTML (LexUZ export) inside .doc extension"
meta["encoding"] = "UTF-8"

# --- chapters / articles / amendments ---------------------------------------
chapters = []
articles = []
amendments = []
annex = {
    "annex_id": "ANNEX_01",
    "banner": None,
    "title_parts": [],
    "title": None,
    "table_id": "ANNEX_TABLE_01",
}

current_chapter = None
current_article = None
in_annex = False
table_el = None

ARTICLE_RE = re.compile(r"^(\d+(?:\s+\d+)?)\s*-\s*modda\.\s*(.*)$", re.I)
CHAPTER_RE = re.compile(r"^(\d+(?:\s+\d+)?)\s*-\s*bob\.\s*(.*)$", re.I)
DOT_PART_RE = re.compile(r"^(\d+(?:\s+\d+)?)\s*\.\s*(.*)$", re.S)
PAREN_PART_RE = re.compile(r"^(\d+(?:\s+\d+)?)\s*\)\s*(.*)$", re.S)
PART_RE = re.compile(r"^(\d+(?:\s+\d+)?)\s*[\)\.]\s*(.*)$", re.S)

AMEND_ACT_RE = re.compile(
    r"(\d{4})-yil\s+(\d{1,2})-([A-Za-zʻʼ\u2018\u2019]+)dagi\s+"
    r"([A-Za-zʻʼ\u2018\u2019]{1,12}\s*-\s*\d+)\s*-\s*sonli"
)


def parse_amendment(raw, context_article):
    inner = raw[1:-1].strip() if raw.startswith("(") and raw.endswith(")") else raw
    ref_m = re.match(r"^\s*(\d+(?:\s+\d+)?)\s*-\s*modda", inner)
    ref = norm_article_num(ref_m.group(1)) if ref_m else None

    acts = []
    for m in AMEND_ACT_RE.finditer(inner):
        acts.append({
            "date": uz_date(m.group(0)),
            "law_number": re.sub(r"\s+", "", m.group(4)),
        })

    eff = re.search(r"[Kk]uchga kirish sanasi\s*[—\-]\s*" + UZ_DATE_RE.pattern, inner)
    effective_date = iso_date(eff.group(1), eff.group(2), eff.group(3)) if eff else None

    pub = re.search(
        r"(Qonunchilik maʼlumotlari milliy bazasi|Qonun hujjatlari maʼlumotlari milliy bazasi),"
        r"\s*([\d\.]+)-y\.,\s*([\d\/]+)-son",
        inner,
    )
    publication = None
    if pub:
        publication = {"database": pub.group(1), "date": pub.group(2), "number": pub.group(3)}

    subject = None
    subj_m = re.search(r"modda(?:ning)?\s+(.*?)\s+Oʻzbekiston Respublikasining", inner)
    if subj_m:
        subject = subj_m.group(1).strip()

    operations = [op for op in (
        "tahririda", "toʻldirilgan", "chiqariladi", "chiqarilgan", "kiritilgan",
        "oʻz kuchini yoʻqotgan", "almashtirilgan", "qoʻshilgan",
    ) if op in inner]

    return {
        "amendment_id": None,  # assigned by add_amendment()
        "article_reference": ref,
        "context_article": context_article,
        "amending_acts": acts,
        "law_number": acts[0]["law_number"] if acts else None,
        "law_date": acts[0]["date"] if acts else None,
        "subject": subject,
        "operations": operations,
        "content": inner,
        "effective_date": effective_date,
        "publication": publication,
        "source_reference": {
            "chapter": current_chapter["chapter_number"] if current_chapter else None,
            "article": context_article,
        },
    }


def add_amendment(raw, context_article):
    amd = parse_amendment(raw, context_article)
    amd["amendment_id"] = f"AMD_{len(amendments) + 1:04d}"
    amendments.append(amd)
    return amd


def finalize_article(art):
    if art is None:
        return
    paragraphs = art.pop("_paragraphs")

    # Some articles (e.g. 28-modda) use "N." for top-level parts and "N)" for
    # sub-items. Others use "N)" for every part. Detect the dominant style so
    # numbering is preserved and never duplicated.
    dot_style = any(DOT_PART_RE.match(p) for p in paragraphs)
    top_re = DOT_PART_RE if dot_style else PAREN_PART_RE
    sub_re = PAREN_PART_RE if dot_style else None

    parts = []
    intro = []
    cur = None
    cur_sub = None

    for p in paragraphs:
        m = top_re.match(p)
        if m:
            cur = {
                "part_number": norm_article_num(m.group(1)),
                "lead_text": norm(m.group(2)),
                "text": norm(m.group(2)),
                "paragraphs": [norm(m.group(2))],
                "subparts": [],
            }
            parts.append(cur)
            cur_sub = None
            continue

        sm = sub_re.match(p) if sub_re else None
        if sm and cur is not None:
            sub = {
                "subpart_number": norm_article_num(sm.group(1)),
                "text": norm(sm.group(2)),
                "paragraphs": [norm(sm.group(2))],
            }
            cur["subparts"].append(sub)
            cur["paragraphs"].append(p)
            cur["text"] = norm(cur["text"] + " " + p)
            cur_sub = sub
            continue

        if cur is not None:
            cur["paragraphs"].append(p)
            cur["text"] = norm(cur["text"] + " " + p)
            if cur_sub is not None:
                cur_sub["paragraphs"].append(p)
                cur_sub["text"] = norm(cur_sub["text"] + " " + p)
        else:
            intro.append(p)

    art["text"] = "\n\n".join(paragraphs)
    art["intro_paragraphs"] = intro
    art["parts"] = parts
    art["paragraph_count"] = len(paragraphs)
    art["part_count"] = len(parts)
    art["part_style"] = "dot" if dot_style else "paren"
    art["amendments"] = art.pop("_amendments")


for el, classes, t in KID_INFO:
    if "APPL_BANNER_LANDSCAPE_TITLE" in classes:
        in_annex = True
        annex["banner"] = t
        if current_article:
            finalize_article(current_article)
            current_article = None
        continue

    if in_annex:
        if "ACT_TITLE_APPL" in classes:
            annex["title_parts"].append(t)
        elif "ACT_FORM" in classes:
            annex["title_parts"].append(t)
        elif "CHANGES_ORIGINS" in classes:
            amd = add_amendment(t, "ILOVA")
            amd["source_reference"] = {"chapter": None, "article": "ILOVA"}
            annex.setdefault("amendments", []).append(amd)
        elif (found := el.find("table")) is not None:
            table_el = found
        continue

    if "TEXT_HEADER_DEFAULT" in classes:
        if current_article:
            finalize_article(current_article)
            current_article = None
        m = CHAPTER_RE.match(t)
        if m:
            num = norm_article_num(m.group(1))
            current_chapter = {
                "chapter_id": f"CHAPTER_{len(chapters) + 1:02d}",
                "chapter_number": num,
                "chapter_title": norm(m.group(2)),
                "articles": [],
            }
            chapters.append(current_chapter)
        continue

    if "CLAUSE_DEFAULT" in classes:
        if current_article:
            finalize_article(current_article)
        m = ARTICLE_RE.match(t)
        if not m:
            continue
        num = norm_article_num(m.group(1))
        current_article = {
            "article_id": article_id(num),
            "article_number": f"{num}-modda",
            "article_number_numeric": num,
            "article_heading_raw": t,
            "article_title": norm(m.group(2)),
            "chapter_id": current_chapter["chapter_id"] if current_chapter else None,
            "chapter_number": current_chapter["chapter_number"] if current_chapter else None,
            "_paragraphs": [],
            "_amendments": [],
            "source_reference": {
                "chapter": f"{current_chapter['chapter_number']}-bob" if current_chapter else None,
                "article": f"{num}-modda",
            },
        }
        if current_chapter:
            current_chapter["articles"].append(current_article)
        articles.append(current_article)
        continue

    if current_article is not None:
        if "ACT_TEXT" in classes:
            if t:
                current_article["_paragraphs"].append(t)
        elif "CHANGES_ORIGINS" in classes:
            amd = add_amendment(t, current_article["article_number"])
            current_article["_amendments"].append(amd)

if current_article:
    finalize_article(current_article)

if annex["title_parts"]:
    annex["title"] = norm(" ".join(annex["title_parts"]))

# --- table extraction -------------------------------------------------------
rows_out = []
sections = []
rate_rows = []
notes = []
current_section = None
current_parent = None
section_counter = 0

if table_el is not None:
    for idx, tr in enumerate(table_el.find_all("tr")):
        cells = tr.find_all(["td", "th"])
        texts = [text_of(c) for c in cells]
        colspans = [c.get("colspan") for c in cells]

        row_type = "unknown"
        section = None
        parent = None
        item = None
        rate = None
        note = None

        if len(texts) == 1:
            value = texts[0]
            if re.match(r"^\d+(?:\s+\d+)?\s*\.", value):
                row_type = "section"
                section = value
                current_section = value
                current_parent = None
                section_counter += 1
                sections.append({"section_index": section_counter, "row_index": idx, "title": value})
            elif value.lower().startswith("izoh"):
                row_type = "note"
                note = value
                notes.append({"row_index": idx, "section": current_section, "parent": current_parent, "text": value})
            else:
                row_type = "parent"
                parent = value
                current_parent = value
        elif len(texts) == 2:
            if not texts[0]:
                if texts[1].lower().startswith("izoh"):
                    row_type = "note"
                    note = texts[1]
                    notes.append({"row_index": idx, "section": current_section, "parent": current_parent, "text": texts[1]})
                else:
                    row_type = "parent"
                    parent = texts[1]
                    current_parent = texts[1]
            elif re.match(r"^\d+(?:\s+\d+)?\s*\.", texts[0]) and not texts[0].lower().startswith("izoh"):
                # A numbered section that carries a single rate in the same row.
                row_type = "section"
                section = texts[0]
                rate = texts[1]
                current_section = section
                current_parent = None
                section_counter += 1
                sections.append({"section_index": section_counter, "row_index": idx, "title": section})
                rate_rows.append({
                    "row_index": idx,
                    "row_type": "rate",
                    "section": section,
                    "parent": None,
                    "item": section,
                    "rate": rate,
                    "note": None,
                    "derived_from_section": True,
                })
            else:
                row_type = "rate"
                item, rate = texts[0], texts[1]
        elif len(texts) == 3:
            row_type = "rate"
            item, rate = texts[1], texts[2]
            current_parent = None  # standalone item
        else:
            row_type = "unknown"

        rec = {
            "row_index": idx,
            "row_type": row_type,
            "cells": texts,
            "colspans": colspans,
            "section": current_section,
            "parent": current_parent if row_type == "rate" else parent,
            "item": item,
            "rate": rate,
            "note": note,
        }
        rows_out.append(rec)
        if row_type == "rate" and item and rate:
            rate_rows.append(rec)

table = {
    "table_id": annex["table_id"],
    "annex_id": annex["annex_id"],
    "title": annex["title"],
    "banner": annex["banner"],
    "context": "OʻRQ-600-son Qonunga ilova",
    "columns": [
        "boʻlim",
        "band / kichik band",
        "stavka",
    ],
    "row_count": len(rows_out),
    "section_count": len(sections),
    "rate_row_count": len(rate_rows),
    "note_count": len(notes),
    "sections": sections,
    "rows": rows_out,
}

annex["table"] = table
annex["section_count"] = len(sections)
annex["rate_row_count"] = len(rate_rows)

# --- assemble structured document ------------------------------------------
law_structured = {
    "metadata": meta,
    "chapters": chapters,
    "tables": [table],
    "annexes": [annex],
    "amendments": amendments,
}

# ---------------------------------------------------------------------------
# 5. Persist processed JSON
# ---------------------------------------------------------------------------

def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


shutil.copy2(SOURCE, RAW / SOURCE.name)

dump_json(PROCESSED / "law_structured.json", law_structured)

articles_out = []
for a in articles:
    articles_out.append({
        "article_id": a["article_id"],
        "article_number": a["article_number"],
        "article_number_numeric": a["article_number_numeric"],
        "article_heading_raw": a["article_heading_raw"],
        "article_title": a["article_title"],
        "chapter_id": a["chapter_id"],
        "chapter_number": a["chapter_number"],
        "text": a["text"],
        "intro_paragraphs": a["intro_paragraphs"],
        "parts": a["parts"],
        "paragraph_count": a["paragraph_count"],
        "part_count": a["part_count"],
        "part_style": a["part_style"],
        "amendments": a["amendments"],
        "source_reference": a["source_reference"],
    })
dump_json(PROCESSED / "articles.json", articles_out)
dump_json(PROCESSED / "tables.json", [table])
dump_json(PROCESSED / "amendments.json", amendments)

# ---------------------------------------------------------------------------
# 6. SFT dataset generation
# ---------------------------------------------------------------------------

CORPUS_PARTS = []
for a in articles_out:
    CORPUS_PARTS.append(a["article_number"])
    CORPUS_PARTS.append(a["article_title"])
    CORPUS_PARTS.extend(a["text"].split("\n\n"))
    for p in a["parts"]:
        CORPUS_PARTS.append(p["text"])
for r in rows_out:
    CORPUS_PARTS.extend([x for x in r["cells"] if x])
CORPUS_PARTS.extend([
    meta.get("law_title") or "", meta.get("law_number") or "",
    date_raw or "", annex.get("title") or "", annex.get("banner") or "",
    "2020-yil 6-yanvar",
])
CORPUS_PARTS.extend(am["content"] for am in amendments)
CORPUS = norm(" ".join(CORPUS_PARTS))


def quotes_ok(quotes):
    """Every quote must occur verbatim in the original document text."""
    return all(norm(q) and norm(q) in FULL_TEXT for q in quotes)


def make_sample(group, task_type, user, assistant, quotes, article=None, chapter=None, extra=None):
    md = {
        "law_number": meta.get("law_number"),
        "law_title": meta.get("law_title"),
        "article": article,
        "chapter": chapter,
        "task_type": task_type,
        "language": "uz",
        "source_reference": {
            "law": meta.get("law_number"),
            "article": article,
            "chapter": chapter,
        },
        "source_quotes": [norm(q) for q in quotes if norm(q)],
        "split_group": group,
    }
    if extra:
        md.update(extra)
    return {
        "messages": [
            {"role": "user", "content": norm(user)},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": md,
    }


raw_samples = []

# Document-level samples
raw_samples.append(make_sample(
    "DOC_METADATA", "legal_qa",
    "Ushbu hujjat qanday nomlanadi, raqami va qabul qilingan sanasi qanday?",
    f"Manbaga koʻra, hujjat «{meta.get('law_title')}»gi Oʻzbekiston Respublikasi Qonuni "
    f"({meta.get('law_number')}-son) hisoblanadi. Qonun {(date_raw or '').rstrip(',')}da qabul qilingan.",
    [meta.get("law_title"), meta.get("law_number"), date_raw],
))

raw_samples.append(make_sample(
    "TABLE_META", "table_understanding",
    "Ilovada davlat boji stavkalari qanday hujjat bilan tasdiqlangan?",
    "Manbadagi ilovaga koʻra, davlat boji stavkalari «Davlat boji toʻgʻrisida»gi "
    "Oʻzbekiston Respublikasi Qonuniga ilova sifatida «Davlat boji stavkalarining MIQDORLARI» "
    "hujjati bilan tasdiqlangan.",
    [annex.get("banner"), annex.get("title"), "MIQDORLARI"],
))

for a in articles_out:
    num = a["article_number"]
    title = a["article_title"]
    text = a["text"]
    paras = a["text"].split("\n\n")
    chapter = f"{a['chapter_number']}-bob"
    group = f"ART_{a['article_number_numeric']}"

    # 9.1 Legal QA
    raw_samples.append(make_sample(
        group, "legal_qa",
        f"{num} nimani tartibga soladi?",
        f"{num} «{title}» masalasini tartibga soladi.\n\nManbadagi normativ matn:\n{text}",
        [title] + paras, article=num, chapter=chapter,
    ))

    # 9.2 Article explanation (explicitly separated from normative text)
    if a["parts"]:
        structure = f"Modda {len(a['parts'])} ta banddan iborat."
    elif a["paragraph_count"] > 1:
        structure = f"Modda {a['paragraph_count']} ta xatboshidan iborat."
    else:
        structure = "Modda bitta xatboshidan iborat."
    raw_samples.append(make_sample(
        group, "article_explanation",
        f"{num}ning mazmunini sodda tilda tushuntir.",
        f"Soddalashtirilgan izoh (bu normativ matn emas): {num} «{title}» masalasini tartibga soladi. "
        f"{structure} Uning normativ mazmuni quyida keltiriladi:\n\n{text}\n\n"
        "Diqqat: yuqoridagi normativ matn manbadan soʻzma-soʻz olingan; izoh faqat modda nomi, "
        "tuzilishi va matniga tayanadi.",
        [title] + paras, article=num, chapter=chapter,
    ))

    # 9.3 Article retrieval
    raw_samples.append(make_sample(
        group, "article_retrieval",
        f"Qaysi modda «{title}» masalasini tartibga soladi?",
        f"Manbada bu masala {num} bilan tartibga solinadi. Modda nomi: «{num}. {title}».",
        [title], article=num, chapter=chapter,
    ))

    # 9.4 Structured extraction
    if a["parts"]:
        lines = []
        marker = "." if a.get("part_style") == "dot" else ")"
        for p in a["parts"]:
            lines.append(f"{p['part_number']}{marker} " + "\n   ".join(p["paragraphs"]))
        raw_samples.append(make_sample(
            group, "structured_extraction",
            f"{num} tarkibidagi barcha bandlarni asl tartibda ajratib ber.",
            f"{num} tarkibida quyidagi bandlar mavjud:\n" + "\n".join(lines),
            [pp for p in a["parts"] for pp in p["paragraphs"]], article=num, chapter=chapter,
            extra={"part_numbers": [p["part_number"] for p in a["parts"]]},
        ))

        # Part-level QA (first / middle / last, to avoid repetitive coverage)
        idxs = sorted({0, len(a["parts"]) // 2, len(a["parts"]) - 1})
        for i in idxs:
            p = a["parts"][i]
            part_text = "\n".join(p["paragraphs"])
            raw_samples.append(make_sample(
                group, "legal_qa",
                f"{num}ning {p['part_number']}-bandida nima belgilangan?",
                f"{num}ning {p['part_number']}-bandida quyidagilar belgilangan:\n{part_text}",
                p["paragraphs"], article=num, chapter=chapter,
                extra={"part_number": p["part_number"]},
            ))
    elif a["paragraph_count"] > 1:
        # No explicit numbered parts: expose the original paragraphs as-is.
        lines = [f"{i}-xatboshi: {p}" for i, p in enumerate(a["intro_paragraphs"], 1)]
        raw_samples.append(make_sample(
            group, "structured_extraction",
            f"{num} qanday xatboshilardan iborat?",
            f"{num} quyidagi xatboshilardan iborat:\n" + "\n".join(lines),
            list(a["intro_paragraphs"]), article=num, chapter=chapter,
        ))

    # Amendment history (separate from normative text)
    if a["amendments"]:
        lines = [f"— {am['content']}" for am in a["amendments"]]
        raw_samples.append(make_sample(
            group, "amendment_history",
            f"{num} boʻyicha qanday oʻzgartirishlar qayd etilgan?",
            f"Manbada {num} boʻyicha quyidagi oʻzgartirishlar qayd etilgan:\n" + "\n".join(lines),
            [am["content"] for am in a["amendments"]], article=num, chapter=chapter,
            extra={"amendment_ids": [am["amendment_id"] for am in a["amendments"]]},
        ))

# 9.5 Table understanding — section summaries
section_rates = OrderedDict()
for r in rate_rows:
    section_rates.setdefault(r["section"], []).append(r)

section_notes = OrderedDict()
for n in notes:
    section_notes.setdefault(n["section"], []).append(n["text"])


def rate_display(rate: str) -> str:
    """Strip only a leading typographic dash used by the source table."""
    return re.sub(r"^[—–-]\s*", "", rate or "").strip()


def note_display(note: str) -> str:
    """Remove the source's 'Izoh .' label from the presentation only."""
    return re.sub(r"^Izoh\s*\.\s*", "", note or "").strip()


covered_notes = set()

for section, rrows in section_rates.items():
    if len(rrows) > 20:
        continue  # very large sections are covered by individual-row samples
    group = table_group(section)
    lines = []
    for r in rrows:
        prefix = f"{r['parent']}: " if r.get("parent") else ""
        lines.append(f"— {prefix}{r['item']} — {rate_display(r['rate'])}")
    sec_notes = section_notes.get(section, [])
    for n in sec_notes:
        lines.append(f"— Izoh: {note_display(n)}")
        covered_notes.add(n)
    raw_samples.append(make_sample(
        group, "table_understanding",
        f"«{section}» boʻyicha davlat boji stavkalari qanday?",
        f"Manbadagi ilovaga koʻra, «{section}» boʻyicha quyidagi stavkalar belgilangan:\n"
        + "\n".join(lines),
        [section] + [x for r in rrows for x in (r.get("parent"), r["item"], r["rate"]) if x] + sec_notes,
        article="ILOVA",
        extra={"table_id": table["table_id"], "table_section": section},
    ))

# Individual rate samples (up to 3 per section)
for section, rrows in section_rates.items():
    group = table_group(section)
    seen = set()
    picked = 0
    for r in rrows:
        label = " — ".join(x for x in (r.get("parent"), r["item"]) if x)
        if label in seen:
            continue
        seen.add(label)
        raw_samples.append(make_sample(
            group, "table_understanding",
            f"«{section}» boʻyicha «{label}» uchun davlat boji stavkasi qanday?",
            f"Manbadagi ilovaga koʻra, «{section}» boʻyicha «{label}» uchun stavka: {rate_display(r['rate'])}.",
            [section, r.get("parent"), r["item"], r["rate"]],
            article="ILOVA",
            extra={"table_id": table["table_id"], "table_section": section, "table_row_index": r["row_index"]},
        ))
        picked += 1
        if picked >= 3:
            break

# Notes in very large sections (or sections without rate rows) still need coverage.
for section, nlist in section_notes.items():
    missing = [n for n in nlist if n not in covered_notes]
    if not missing:
        continue
    group = table_group(section)
    raw_samples.append(make_sample(
        group, "table_understanding",
        f"«{section}» boʻyicha ilovadagi izohlarni keltir.",
        f"Manbadagi ilovaga koʻra, «{section}» boʻyicha quyidagi izohlar keltirilgan:\n"
        + "\n".join(f"— {note_display(n)}" for n in missing),
        [section] + missing,
        article="ILOVA",
        extra={"table_id": table["table_id"], "table_section": section},
    ))

# 9.6 Insufficient evidence
INSUFFICIENT = "Berilgan hujjatda ushbu savolga javob berish uchun yetarli maʼlumot topilmadi."
for i, q in enumerate([
    "Ushbu hujjatda jinoyat sodir etganlik uchun jazo turlari qanday belgilangan?",
    "Ushbu hujjatda nikoh tuzish tartibi qanday?",
    "Ushbu hujjatda yer soligʻi stavkalari qanday belgilangan?",
    "Ushbu hujjatda mehnat shartnomasini bekor qilish tartibi qanday?",
    "Ushbu hujjatda saylov oʻtkazish tartibi qanday?",
    "Ushbu hujjatda avtomobil sugʻurtasi boʻyicha tariflar qanday?",
], 1):
    raw_samples.append(make_sample(
        f"INSUFFICIENT_{i:02d}", "insufficient_evidence", q, INSUFFICIENT, [],
    ))

# ---------------------------------------------------------------------------
# 7. Deduplicate + source verification + grouped split
# ---------------------------------------------------------------------------

seen_pairs = set()
samples = []
review_required = []
duplicates_removed = 0

for s in raw_samples:
    key = (s["messages"][0]["content"], s["messages"][1]["content"])
    if key in seen_pairs:
        duplicates_removed += 1
        continue
    seen_pairs.add(key)

    md = s["metadata"]
    quotes = md.get("source_quotes") or []
    if quotes and not quotes_ok(quotes):
        md["review_reason"] = "source_quote not found verbatim in the original document"
        review_required.append(s)
        continue
    if not quotes and md.get("task_type") != "insufficient_evidence":
        md["review_reason"] = "no source quote to verify"
        review_required.append(s)
        continue
    samples.append(s)

# Review-only records get their own IDs and are kept out of train/val/test.
for i, s in enumerate(review_required, 1):
    s["metadata"]["sample_id"] = f"REV-{i:04d}"
    s["metadata"]["split"] = "review_required"

group_sizes = Counter(s["metadata"]["split_group"] for s in samples)
groups = list(group_sizes)

TARGET = {"train": 0.80, "validation": 0.10, "test": 0.10}
group_split = {}


def assign_category(cat_groups):
    """Stratified grouped split inside one category (articles / table sections)."""
    cur = {"train": 0, "validation": 0, "test": 0}
    cat_total = sum(group_sizes[g] for g in cat_groups)
    for g in sorted(cat_groups, key=lambda x: (-group_sizes[x], x)):
        deficits = {sp: TARGET[sp] * cat_total - cur[sp] for sp in TARGET}
        sp = max(deficits, key=deficits.get)
        group_split[g] = sp
        cur[sp] += group_sizes[g]
    return cur


article_groups = [g for g in groups if g.startswith("ART_")]
table_groups = [g for g in groups if g.startswith("TBL_")]
other_groups = [g for g in groups if g not in article_groups and g not in table_groups]
insufficient_groups = sorted(g for g in other_groups if g.startswith("INSUFFICIENT"))
doc_groups = [g for g in other_groups if g not in insufficient_groups]

assign_category(article_groups)
assign_category(table_groups)
for g in doc_groups:
    group_split[g] = "train"  # document metadata / table metadata examples

# Refusal examples: mostly train, with one each in validation and test.
for i, g in enumerate(insufficient_groups):
    if i < len(insufficient_groups) - 2:
        group_split[g] = "train"
    elif i == len(insufficient_groups) - 2:
        group_split[g] = "validation"
    else:
        group_split[g] = "test"

# Ensure validation and test contain both article and table groups.
for cat in (article_groups, table_groups):
    for sp in ("validation", "test"):
        if not any(group_split[g] == sp for g in cat):
            movable = [g for g in cat if group_split[g] == "train"]
            if movable:
                group_split[min(movable, key=lambda x: group_sizes[x])] = sp

for s in samples:
    s["metadata"]["split"] = group_split[s["metadata"]["split_group"]]

splits = {"train": [], "validation": [], "test": []}
for s in samples:
    splits[s["metadata"]["split"]].append(s)

# Deterministic ordering inside each split, then stable IDs.
for sp, items in splits.items():
    items.sort(key=lambda x: (x["metadata"]["split_group"], x["metadata"]["task_type"], x["messages"][0]["content"]))
    for i, s in enumerate(items, 1):
        s["metadata"]["sample_id"] = f"SFT-{sp[:3].upper()}-{i:04d}"

# ---------------------------------------------------------------------------
# 8. Validation
# ---------------------------------------------------------------------------

errors = []
warnings = []

article_numbers = {a["article_number"] for a in articles_out} | {"ILOVA"}
article_numeric = {a["article_number_numeric"] for a in articles_out}
sample_ids = set()
pair_keys = set()
group_to_split = {}
split_groups = {sp: set() for sp in splits}

for sp, items in splits.items():
    for s in items:
        sid = s["metadata"]["sample_id"]
        if sid in sample_ids:
            errors.append(f"Duplicate sample_id: {sid}")
        sample_ids.add(sid)

        msgs = s.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 2:
            errors.append(f"{sid}: messages must contain exactly 2 entries")
            continue
        if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            errors.append(f"{sid}: invalid role order")
        if not norm(msgs[0].get("content", "")) or not norm(msgs[1].get("content", "")):
            errors.append(f"{sid}: empty message content")

        md = s.get("metadata", {})
        required = ["law_number", "task_type", "language", "source_reference", "source_quotes", "split_group", "split"]
        for field in required:
            if field == "source_quotes" and md.get("task_type") == "insufficient_evidence":
                continue
            if field not in md or md[field] in (None, "", []):
                errors.append(f"{sid}: missing metadata.{field}")

        art = md.get("article")
        if art and art not in article_numbers and md.get("task_type") != "insufficient_evidence":
            errors.append(f"{sid}: unknown article reference {art}")

        g = md.get("split_group")
        if g in group_to_split and group_to_split[g] != sp:
            errors.append(f"{sid}: data leakage, group {g} in both {group_to_split[g]} and {sp}")
        group_to_split[g] = sp
        split_groups[sp].add(g)

        key = (msgs[0]["content"], msgs[1]["content"])
        if key in pair_keys:
            errors.append(f"{sid}: duplicate question/answer pair")
        pair_keys.add(key)

# Verify article numbering against source headings
for a in articles_out:
    if not a["article_number"].endswith("-modda"):
        errors.append(f"Bad article number: {a['article_number']}")

# Unicode sanity
for ch in ("oʻ", "gʻ", "ʻ"):
    if ch not in CORPUS:
        warnings.append(f"Expected Uzbek character not found in corpus: {ch!r}")

# ---------------------------------------------------------------------------
# Write dataset files
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for s in items:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


write_jsonl(DATASET / "train.jsonl", splits["train"])
write_jsonl(DATASET / "validation.jsonl", splits["validation"])
write_jsonl(DATASET / "test.jsonl", splits["test"])
write_jsonl(DATASET / "review_required.jsonl", review_required)

# Read-back validation: every written file must parse again.
readback_ok = True
readback = {}
for p in (
    PROCESSED / "law_structured.json",
    PROCESSED / "articles.json",
    PROCESSED / "tables.json",
    PROCESSED / "amendments.json",
):
    try:
        json.loads(p.read_text(encoding="utf-8"))
        readback[p.name] = "ok"
    except Exception as exc:  # pragma: no cover
        readback_ok = False
        readback[p.name] = f"error: {exc}"
        errors.append(f"{p.name}: read-back JSON parse failed: {exc}")

for p in (
    DATASET / "train.jsonl",
    DATASET / "validation.jsonl",
    DATASET / "test.jsonl",
    DATASET / "review_required.jsonl",
):
    n = 0
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
                    n += 1
        readback[p.name] = {"status": "ok", "records": n}
    except Exception as exc:  # pragma: no cover
        readback_ok = False
        readback[p.name] = f"error: {exc}"
        errors.append(f"{p.name}: read-back JSONL parse failed: {exc}")

# ---------------------------------------------------------------------------
# 9. Reports
# ---------------------------------------------------------------------------

extraction_report = {
    "source_file": str(SOURCE.relative_to(ROOT)),
    "detected_format": meta["detected_format"],
    "encoding": meta["encoding"],
    "metadata": meta,
    "counts": {
        "chapters": len(chapters),
        "articles": len(articles_out),
        "tables": 1 if table_el is not None else 0,
        "annexes": 1,
        "amendments": len(amendments),
        "table_rows": len(rows_out),
        "table_sections": len(sections),
        "table_rate_rows": len(rate_rows),
        "table_notes": len(notes),
    },
    "chapters": [
        {
            "chapter_id": c["chapter_id"],
            "chapter_number": c["chapter_number"],
            "chapter_title": c["chapter_title"],
            "article_count": len(c["articles"]),
        }
        for c in chapters
    ],
    "articles": [
        {
            "article_number": a["article_number"],
            "article_title": a["article_title"],
            "chapter_number": a["chapter_number"],
            "paragraph_count": a["paragraph_count"],
            "part_count": a["part_count"],
            "amendment_count": len(a["amendments"]),
        }
        for a in articles_out
    ],
    "warnings": warnings,
    "notes": [
        "Manba .doc kengaytmali boʻlsa-da, ichki formati HTML (LexUZ eksporti), UTF-8.",
        "Hujjat tartibi divCont ichidagi elementlar ketma-ketligi boʻyicha saqlandi.",
        "Ilovadagi boʻlim raqamlari manbada uzluksiz emas; barcha mavjud boʻlimlar oʻzgartirilmasdan saqlandi.",
        f"Oʻzgartirishlar tarixi ({len(amendments)} ta yozuv) normativ matndan alohida saqlandi.",
        f"Jadvalning {len(rows_out)} satri, {len(rate_rows)} stavka satri va {len(notes)} izohi saqlandi.",
    ],
}

article_headings_ok = (
    len(SOURCE_ARTICLE_HEADINGS) == len(articles_out)
    and all(norm(src) == a["article_heading_raw"] for src, a in zip(SOURCE_ARTICLE_HEADINGS, articles_out))
)

quality_report = {
    "checks": {
        "json_parse": "ok" if readback_ok else "failed",
        "jsonl_parse": "ok" if readback_ok else "failed",
        "required_fields": "ok" if not any("missing metadata" in e for e in errors) else "failed",
        "unicode_preserved": all(ch in CORPUS for ch in ("oʻ", "gʻ")),
        "empty_records": "ok" if not any("empty message" in e for e in errors) else "failed",
        "duplicate_ids": "ok" if not any("Duplicate sample_id" in e for e in errors) else "failed",
        "article_numbers_match_source": "ok" if article_headings_ok else "failed",
        "source_quotes_verified": "ok" if not review_required else "review_required",
        "grouped_split_no_leakage": "ok" if not any("data leakage" in e for e in errors) else "failed",
        "no_duplicate_qa_pairs": "ok" if not any("duplicate question" in e for e in errors) else "failed",
    },
    "readback": readback,
    "errors": errors,
    "warnings": warnings,
    "review_required_count": len(review_required),
    "review_required_reasons": dict(Counter(
        r["metadata"].get("review_reason", "unknown") for r in review_required
    )),
    "notes": [
        "Normativ matn, raqamlar, sanalar va jadval qiymatlari manbadan soʻzma-soʻz olingan.",
        "Oʻzgartirishlar tarixi normativ matndan alohida saqlangan (articles.json, amendments.json).",
        "Javoblar manba iqtiboslari (source_quotes) bilan tekshirilgan.",
        "Jadvalning 1 ustunli qatorlari boʻlim yoki izoh sifatida, 2 ustunli boʻsh birinchi katakli qatorlar esa band tavsifi sifatida talqin qilindi.",
    ],
}

task_counts = Counter(s["metadata"]["task_type"] for s in samples)
split_counts = {sp: len(items) for sp, items in splits.items()}
dataset_statistics = {
    "total_samples": len(samples),
    "total_generated_samples": len(raw_samples),
    "by_split": split_counts,
    "by_task_type": dict(task_counts),
    "by_task_type_per_split": {
        sp: dict(Counter(s["metadata"]["task_type"] for s in items))
        for sp, items in splits.items()
    },
    "groups": {
        "total": len(group_sizes),
        "train": len(split_groups["train"]),
        "validation": len(split_groups["validation"]),
        "test": len(split_groups["test"]),
    },
    "articles_total": len(articles_out),
    "articles_covered_by_samples": len({
        s["metadata"]["article"] for s in samples
        if s["metadata"].get("article") and s["metadata"]["article"] != "ILOVA"
    }),
    "table_sections_total": len(sections),
    "table_sections_covered_by_samples": len({
        s["metadata"].get("table_section") for s in samples if s["metadata"].get("table_section")
    }),
    "duplicates_removed": duplicates_removed,
    "review_required": len(review_required),
    "split_ratio_target": TARGET,
    "split_ratio_actual": {sp: round(n / len(samples), 4) for sp, n in split_counts.items()},
    "strategy_note": (
        "Split modda va jadval boʻlimi guruhlari boʻyicha amalga oshirildi; "
        "bir moddaning barcha namunalari bitta toʻplamda qoladi (data leakage oldini olish)."
    ),
}

dump_json(REPORTS / "extraction_report.json", extraction_report)
dump_json(REPORTS / "quality_report.json", quality_report)
dump_json(REPORTS / "dataset_statistics.json", dataset_statistics)

summary_md = f"""# LexUZ hujjatini qayta ishlash hisoboti

## Manba
- Fayl: `{extraction_report['source_file']}`
- Aniqlangan format: {meta['detected_format']}
- Kodlash: {meta['encoding']}
- Qonun: **{meta.get('law_title')}**
- Raqam: **{meta.get('law_number')}**
- Qabul qilingan sana: **{meta.get('adoption_date')}**

## Ajratilgan tarkib
| Element | Soni |
|---|---:|
| Boblar | {extraction_report['counts']['chapters']} |
| Moddalar | {extraction_report['counts']['articles']} |
| Jadvallar | {extraction_report['counts']['tables']} |
| Ilovalar | {extraction_report['counts']['annexes']} |
| Oʻzgartirishlar tarixi | {extraction_report['counts']['amendments']} |
| Jadval satrlari | {extraction_report['counts']['table_rows']} |
| Jadval boʻlimlari | {extraction_report['counts']['table_sections']} |
| Jadval stavka satrlari | {extraction_report['counts']['table_rate_rows']} |

## Dataset
- Jami tekshirilgan namunalar: **{len(samples)}**
- Train: **{split_counts['train']}**
- Validation: **{split_counts['validation']}**
- Test: **{split_counts['test']}**
- Tekshirish talab qiladigan yozuvlar: **{len(review_required)}**

### Topshiriq turlari
| Tur | Soni |
|---|---:|
""" + "\n".join(f"| {k} | {v} |" for k, v in sorted(task_counts.items())) + f"""

## Sifat tekshiruvi
- JSON/JSONL qayta oʻqish: **{'ok' if readback_ok else 'failed'}**
- Majburiy maydonlar: **{quality_report['checks']['required_fields']}**
- Manba iqtiboslari: **{'ok' if not review_required else 'review_required'}**
- Guruhlar boʻyicha split (leakage yoʻq): **{'ok' if not any('data leakage' in e for e in errors) else 'failed'}**
- Takroriy ID / savol-javob: **{'ok' if not any('Duplicate sample_id' in e or 'duplicate question' in e for e in errors) else 'failed'}**

## Izohlar
""" + "\n".join(f"- {n}" for n in quality_report["notes"]) + "\n"

(REPORTS / "summary.md").write_text(summary_md, encoding="utf-8")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("=== EXTRACTION ===")
print(json.dumps(extraction_report["counts"], ensure_ascii=False, indent=2))
print("law:", meta.get("law_title"), "|", meta.get("law_number"), "|", meta.get("adoption_date"))
print("\n=== DATASET ===")
print("total:", len(samples), "| splits:", split_counts)
print("task types:", dict(task_counts))
print("duplicates removed:", duplicates_removed, "| review required:", len(review_required))
print("errors:", len(errors))
for e in errors[:20]:
    print("  ERROR:", e)
print("warnings:", len(warnings))
for w in warnings[:10]:
    print("  WARN:", w)
print("\nOutputs:")
for p in [
    RAW / SOURCE.name,
    PROCESSED / "law_structured.json",
    PROCESSED / "articles.json",
    PROCESSED / "tables.json",
    PROCESSED / "amendments.json",
    DATASET / "train.jsonl",
    DATASET / "validation.jsonl",
    DATASET / "test.jsonl",
    DATASET / "review_required.jsonl",
    REPORTS / "extraction_report.json",
    REPORTS / "quality_report.json",
    REPORTS / "dataset_statistics.json",
    REPORTS / "summary.md",
]:
    print(" ", p.relative_to(ROOT), p.stat().st_size, "bytes")
