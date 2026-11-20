# VECTOR_INDEXING_AGENT.md
# Legal Document Vector Indexing Agent

## 1. Agentning vazifasi

Sen Legal NLP, Information Retrieval va Vector Database yo‘nalishlariga ixtisoslashgan AI-Agent sifatida ishlaysan.

Asosiy vazifang — Data Preparation bosqichida tayyorlangan strukturali huquqiy JSON fayllarini qayta ishlab, RAG (Retrieval-Augmented Generation) tizimi uchun vektor indeks yaratish.

Loyiha konteksti:

- Domain: O‘zbekiston Respublikasi qonunchiligi
- Language: Uzbek
- Primary source: LexUZ
- Target LLM: Gemma 4
- Embedding model: intfloat/multilingual-e5-large
- Vector database: FAISS
- Keyword search: BM25
- Retrieval strategy: Hybrid Search
- Architecture: RAG

Agentning yakuniy maqsadi — foydalanuvchi savoliga mos qonun moddasi, qismi, bandi yoki jadval ma'lumotini tez va aniq topish.

## 2. Asosiy vazifalar

Agent quyidagi pipeline'ni bajarishi kerak:

1. Structured JSON faylni yuklash.
2. JSON ma'lumotlarini tekshirish.
3. Huquqiy matnlarni semantik chunklarga ajratish.
4. Metadata biriktirish.
5. Embedding yaratish.
6. Vector index yaratish.
7. BM25 indeks yaratish.
8. Hybrid retrieval tizimini yaratish.
9. Retrieval natijalarini tekshirish.
10. Index va metadata fayllarini saqlash.
11. Indexing report yaratish.

## 3. Input Data

Asosiy kirish fayli:

`data/processed/law_structured.json`

Qo‘shimcha fayllar:

- `articles.json`
- `tables.json`
- `amendments.json`

Agar fayllar mavjud bo‘lmasa, ularni o‘zingdan yaratma. Mavjud fayllardan foydalan va yetishmayotgan ma'lumotlarni reportda qayd et.

Input JSON quyidagi ma'lumotlarni o‘z ichiga olishi mumkin:

- Qonun nomi.
- Qonun raqami.
- Boblar.
- Moddalar.
- Modda qismlari.
- Bandlar.
- Normativ matn.
- Jadvallar.
- Ilovalar.
- O‘zgartirishlar tarixi.
- Source reference.

## 4. Data Validation

Indexing boshlanishidan oldin JSON faylni tekshir.

Quyidagi holatlarni aniqlash kerak:

- JSON sintaksisi to‘g‘rimi?
- Modda ID'lari mavjudmi?
- Modda matnlari bo‘shmi?
- Takroriy ID'lar bormi?
- Modda raqamlari saqlanganmi?
- Metadata yetarlimi?
- Jadval ma'lumotlari to‘liqmi?

Xatolik aniqlansa, asl JSON faylni o‘zgartirma.

Xatolarni quyidagi faylga yoz:

`data/reports/indexing_validation_report.json`

## 5. Text Chunking

Huquqiy hujjatlarni vektor indekslashdan oldin chunklarga ajrat.

Chunking strategiyasi:

**Structure-Aware Chunking**

Oddiy belgilar soniga qarab matnni kesma. Avval qonunning strukturaviy elementlarini hisobga ol:

1. Qonun.
2. Bob.
3. Modda.
4. Modda qismi.
5. Band.
6. Kichik band.

### Chunking qoidalari

- Har bir chunk mustaqil ma'noga ega bo‘lishi kerak.
- Modda raqamini chunk ichida saqla.
- Bob nomini metadata orqali biriktir.
- Qism va band raqamlarini saqla.
- Huquqiy jumlani o‘rtasidan kesma.
- Bir nechta moddaning matnini bitta chunkka birlashtirma.
- Juda katta moddalarni qismlar bo‘yicha ajrat.
- Kichik moddalarni zarurat bo‘lsa bitta chunk sifatida saqla.
- Jadval qatorlarini tegishli sarlavha va ustun nomlari bilan birga saqla.

### Tavsiya etilgan chunk hajmi

Boshlang‘ich konfiguratsiya:

- Target chunk size: 400–700 tokens.
- Maximum chunk size: 900 tokens.
- Overlap: 50–100 tokens.
- Minimum chunk size: 80 tokens.

Bu qiymatlar boshlang‘ich parametrlar hisoblanadi. Ularni real hujjat va retrieval natijalariga qarab optimallashtir.

Muhim: chunking uchun token sonini embedding modelining tokenizer'i yordamida hisobla.

### Katta moddalarni ajratish

Agar modda bir nechta mustaqil qismdan iborat bo‘lsa:

Har bir qism alohida chunk bo‘lishi mumkin.

Masalan:

- 5-modda, 1-qism
- 5-modda, 2-qism
- 5-modda, 3-qism

Har bir chunkda qonun nomi, modda raqami va tegishli qism raqami saqlanishi kerak.

Agar qismni alohida olish uning ma'nosini yo‘qotsa, qo‘shni kontekstni saqla.

## 6. Metadata Design

Har bir chunk uchun metadata yarat.

Tavsiya etilgan schema:

    {
      "chunk_id": "ORQ600_ART_001_CHUNK_001",
      "law_title": "Davlat boji to‘g‘risida",
      "law_number": "O‘RQ-600",
      "chapter": "1-bob",
      "article": "1-modda",
      "part": null,
      "paragraph": null,
      "chunk_type": "article",
      "language": "uz",
      "source": "LexUZ",
      "source_reference": "O‘RQ-600, 1-modda",
      "document_version": null,
      "effective_from": null,
      "text": "Chunkning normativ matni..."
    }

Metadata qoidalari:

- `chunk_id` takrorlanmas bo‘lishi kerak.
- `article` majburiy metadata hisoblanadi, agar chunk modda tarkibiga tegishli bo‘lsa.
- `part` va `paragraph` mavjud bo‘lsa to‘ldiriladi.
- `chunk_type` qiymati article, table, annex yoki other kabi kategoriyalardan biri bo‘lishi mumkin.
- `effective_from` faqat manbada tasdiqlangan bo‘lsa to‘ldiriladi.
- Noma'lum ma'lumotni taxmin qilib yozma.
- Metadata va embedding matni bir-biriga mos bo‘lishi kerak.

## 7. Embedding Generation

Embedding yaratish uchun quyidagi modeldan foydalan:

`intfloat/multilingual-e5-large`

Python kutubxonalari:

    pip install sentence-transformers
    pip install transformers
    pip install torch
    pip install faiss-cpu
    pip install rank-bm25
    pip install numpy

Modelni yuklash:

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large"
    )

### Passage Embedding

Hujjat chunklari uchun `passage:` prefiksidan foydalan.

    passages = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

### Query Embedding

Foydalanuvchi savoli uchun `query:` prefiksidan foydalan.

    query = "query: Davlat boji nima?"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

Muhim qoidalar:

- Query va passage prefikslarini adashtirma.
- Embeddinglarni normalize qil.
- Modelning embedding dimension qiymatini avtomatik aniqlagin.
- GPU mavjud bo‘lsa undan foydalan.
- GPU bo‘lmasa CPU orqali ishlashni ta'minla.
- Batch processing ishlat.
- Bo‘sh matn uchun embedding yaratma.

## 8. Vector Index — FAISS

Vektorlarni saqlash uchun FAISS kutubxonasidan foydalan.

Boshlang‘ich konfiguratsiya:

- Index type: IndexFlatIP
- Similarity: Cosine similarity
- Vector normalization: Enabled

Normalize qilingan vektorlar uchun Inner Product cosine similarity bilan ekvivalent natija beradi.

Namuna:

    import faiss
    import numpy as np

    vectors = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    scores, indices = index.search(
        query_embedding.astype("float32"),
        k=5
    )

Har bir vektorning tartibi metadata faylidagi mos chunk bilan bir xil bo‘lishi shart.

Index yaratishda:

- Chunk ID'larni alohida saqla.
- Metadata tartibini saqla.
- Vektorlar va metadata sonini solishtir.
- Takroriy chunklarni tekshir.
- Indexni qayta yuklash imkoniyatini ta'minla.

## 9. BM25 Keyword Index

Faqat vektor qidiruviga tayanma.

Huquqiy hujjatlarda quyidagi ma'lumotlar juda muhim:

- Modda raqamlari.
- Qonun raqamlari.
- BHM qiymatlari.
- Foizlar.
- Pul miqdorlari.
- Sana va muddatlar.
- Huquqiy terminlar.

Shu sababli BM25 keyword search tizimini ham yarat.

BM25 uchun:

- Matnlarni tokenlarga ajrat.
- O‘zbek tilidagi maxsus belgilarni saqla.
- Raqamlarni yo‘qotma.
- Modda va band raqamlarini saqla.
- Tokenization natijalarini alohida saqla.

Lexical qidiruv va semantik qidiruv bir-birini to‘ldirishi kerak.

## 10. Hybrid Search

Foydalanuvchi savolini ikkita usulda qidir:

1. FAISS orqali semantic search.
2. BM25 orqali keyword search.

Natijalarni birlashtirish uchun Reciprocal Rank Fusion (RRF) usulidan foydalan.

RRF formulasi:

    RRF(d) = sum(
        1 / (k + rank_i(d))
    )

Boshlang‘ich `k` qiymati: 60.

Har bir qidiruv natijasining reytingini hisobla va umumiy reyting bo‘yicha sarala.

Agar foydalanuvchi aniq modda raqamini so‘rasa, exact metadata filter yoki lexical search orqali shu moddaga ustuvorlik ber.

Hybrid Search natijasi quyidagi ma'lumotlarni qaytarishi kerak:

- Chunk ID.
- Modda raqami.
- Chunk matni.
- Similarity score.
- BM25 score yoki rank.
- Final RRF score.
- Source reference.

## 11. Retrieval Strategy

Qidiruv tizimi quyidagi holatlarni qo‘llab-quvvatlashi kerak:

### Semantic Search

Misol:

"Fuqarolar davlat boji qaysi holatlarda to‘laydi?"

Mazmunan mos huquqiy chunklarni top.

### Exact Article Search

Misol:

"O‘RQ-600 qonunining 5-moddasi"

Aniq 5-modda bilan bog‘liq chunklarni top.

### Numeric Search

Misol:

"Davlat boji stavkasi qancha?"

Raqamli qiymatlar, jadval qatorlari va o‘lchov birliklarini saqla.

### Table Retrieval

Jadvalga tegishli savollarda:

- Jadval nomini top.
- Ustun nomlarini saqla.
- Tegishli qatorni top.
- Qiymat va o‘lchov birliklarini birga qaytar.

### Multi-Article Retrieval

Savol bir nechta moddani talab qilsa, barcha tegishli moddalarni qidir.

## 12. Retrieval Quality Evaluation

Index yaratilgandan keyin retrieval sifatini tekshir.

Test savollarini qonun matnidan kelib chiqib tayyorla.

Masalan:

- 1-modda nimani belgilaydi?
- Davlat boji qaysi holatlarda undiriladi?
- Tegishli davlat boji stavkasi qanday?
- Qaysi modda ma'lum bir masalani tartibga soladi?

Har bir test savol uchun kutilgan modda yoki chunk ID'ni belgilab qo‘y.

Quyidagi metrikalarni hisobla:

- Precision@K
- Recall@K
- MRR
- nDCG@K
- Hit Rate@K

Boshlang‘ich sinov uchun K = 5 va K = 10 qiymatlaridan foydalan.

Faqat umumiy similarity score'ga qarab tizim sifatli deb xulosa qilma. Haqiqiy relevant chunk topilganini tekshir.

## 13. RAG Context Preparation

Retrieval natijalarini Gemma 4 modeliga yuborishga tayyorla.

Har bir topilgan chunk uchun:

- Qonun nomi.
- Qonun raqami.
- Modda raqami.
- Qism yoki band.
- Normativ matn.
- Manba ma'lumotini saqla.

Gemma 4 ga yuboriladigan context quyidagi shaklda bo‘lishi mumkin:

    [SOURCE 1]
    Law: Davlat boji to‘g‘risida
    Law Number: O‘RQ-600
    Article: 1-modda
    Text: ...

    [SOURCE 2]
    Law: Davlat boji to‘g‘risida
    Law Number: O‘RQ-600
    Article: 2-modda
    Text: ...

Modelga quyidagi qoidani ber:

"Javobni faqat taqdim etilgan huquqiy manbalar asosida shakllantir. Tegishli modda raqamini ko‘rsat. Manbada javob topilmasa, bu haqda ochiq ayt. Huquqiy faktlarni taxmin qilib yaratma."

## 14. File Structure

Barcha natijalarni quyidagi strukturada saqla:

    vector_index/
    ├── embeddings/
    │   └── law_embeddings.npy
    │
    ├── faiss/
    │   └── law_index.faiss
    │
    ├── bm25/
    │   └── bm25_index.pkl
    │
    ├── metadata/
    │   ├── chunks.json
    │   └── chunk_mapping.json
    │
    ├── config/
    │   └── index_config.json
    │
    └── reports/
        ├── indexing_report.json
        └── retrieval_evaluation.json

`index_config.json` ichida quyidagilarni saqla:

- Embedding model nomi.
- Model versiyasi yoki revision.
- Embedding dimension.
- Vector index turi.
- Similarity metric.
- Chunk size.
- Overlap.
- Chunking strategiyasi.
- Dataset versiyasi.
- Yaratilgan sana.

## 15. Reproducibility va Index Management

Indexni qayta yaratish imkoniyati bo‘lishi kerak.

Agent quyidagilarni ta'minlasin:

- Bir xil input va konfiguratsiyadan qayta tiklanadigan index.
- Chunk ID orqali metadata topish.
- Index va metadata versiyalarini birga boshqarish.
- Eski indexni yangisi bilan adashtirmaslik.
- Fayllarni saqlashdan oldin tekshirish.
- Index fayllarini xavfsiz saqlash.

Agar embedding modeli o‘zgarsa, eski embeddinglar bilan yangi embeddinglarni bitta indeksda aralashtirma. Indexni qayta yarat.

## 16. Final Validation

Yakuniy tekshiruv:

1. FAISS index ichidagi vektorlar soni chunklar soniga tengmi?
2. Har bir vektor uchun metadata mavjudmi?
3. Embedding dimension to‘g‘rimi?
4. Vektorlar normalize qilinganmi?
5. BM25 index ishlayaptimi?
6. Query embedding to‘g‘ri prefiks bilan yaratilayaptimi?
7. Exact article search ishlayaptimi?
8. Numeric search ishlayaptimi?
9. Jadval ma'lumotlari topilyaptimi?
10. Retrieval natijalari source reference bilan qaytyaptimi?

Xatolarni `indexing_report.json` fayliga yoz.

## 17. Final Output

Agent quyidagi natijalarni yaratishi shart:

- `law_index.faiss`
- `law_embeddings.npy`
- `bm25_index.pkl`
- `chunks.json`
- `chunk_mapping.json`
- `index_config.json`
- `indexing_report.json`
- `retrieval_evaluation.json`

Agent ishni yakunlaganda quyidagilarni hisobotda ko‘rsatsin:

- Qayta ishlangan hujjatlar soni.
- Ajratilgan chunklar soni.
- O‘rtacha chunk hajmi.
- Embedding modeli.
- Embedding dimension.
- FAISS index turi.
- BM25 index holati.
- Retrieval test natijalari.
- Aniqlangan xatolar.

## 18. Eng muhim qoidalar

1. Huquqiy matnni o‘zgartirma.
2. Modda raqamini yo‘qotma.
3. Jadval qiymatlarini ajratib yuborma.
4. Metadata va embedding o‘rtasidagi moslikni saqla.
5. Faqat semantik qidiruv bilan cheklanma.
6. Hybrid Search ishlat.
7. Retrieval natijasini manba bilan qaytar.
8. Noaniq yoki topilmagan ma'lumotni to‘qib chiqarmagin.
9. Har bir index uchun konfiguratsiya va versiya saqla.
10. Retrieval sifatini real test savollari bilan bahola.

**Asosiy maqsad:** Gemma 4 uchun ishonchli, manbaga bog‘langan, o‘zbek tilidagi huquqiy RAG retrieval tizimini yaratish.