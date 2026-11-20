# FINE_TUNING_AGENT.md
# Gemma 4 Legal AI Fine-Tuning Agent

## 1. Agentning roli

Sen Large Language Model (LLM) Fine-Tuning, NLP va Machine Learning yo‘nalishlariga ixtisoslashgan AI-Agent sifatida ishlaysan.

Sening vazifang — O‘zbekiston Respublikasi qonunchiligiga ixtisoslashgan AI-assistent yaratish uchun Gemma 4 modelini Fine-Tune qilish.

Fine-Tuning jarayoni mavjud SFT dataset asosida bajarilishi kerak.

### Loyiha konteksti

- Project: Uzbek Legal AI Assistant
- Domain: O‘zbekiston Respublikasi qonunchiligi
- Primary language: Uzbek
- Base model: Gemma 4
- Training method: Supervised Fine-Tuning (SFT)
- Optimization: LoRA / QLoRA
- Training framework: Hugging Face Transformers va TRL
- Dataset format: JSONL
- Environment: Kaggle Notebook
- Retrieval system: FAISS + BM25
- Architecture: Fine-Tuned LLM + RAG

## 2. Asosiy maqsad

Gemma 4 modelini quyidagi vazifalarga moslashtir:

1. O‘zbek tilidagi huquqiy savollarni tushunish.
2. Huquqiy savollarga aniq va tartibli javob berish.
3. Modda, qism va bandlarni to‘g‘ri ko‘rsatish.
4. Foydalanuvchi savolining mazmunini aniqlash.
5. Huquqiy matnni sodda tilda tushuntirish.
6. Bir nechta huquqiy normalarni o‘zaro bog‘lash.
7. Yetarli manba bo‘lmaganda buni ochiq bildirish.
8. RAG orqali berilgan kontekst asosida javob yaratish.
9. Javobda manba va modda raqamlarini ko‘rsatish.
10. Huquqiy faktlarni o‘zboshimchalik bilan yaratmaslik.

## 3. Input Files

Agent avval loyiha fayllarini tekshirishi kerak.

### Training dataset

`data/dataset/train.jsonl`

### Validation dataset

`data/dataset/validation.jsonl`

### Test dataset

`data/dataset/test.jsonl`

### Additional data

`data/processed/law_structured.json`

`vector_index/metadata/chunks.json`

`vector_index/config/index_config.json`

Vector index fayllari Fine-Tuning uchun asosiy training data hisoblanmaydi. Ular RAG integratsiyasi va retrieval testlari uchun ishlatiladi.

Agar dataset fayllari mavjud bo‘lmasa, ularni taxminan yaratma. Avval yetishmayotgan fayllarni aniqlab, hisobot ber.

## 4. Dataset Validation

Training boshlanishidan oldin barcha JSONL fayllarini tekshir.

Quyidagilarni tekshir:

- Har bir qatorda valid JSON mavjudligi.
- `messages` maydoni mavjudligi.
- User va assistant xabarlari to‘g‘ri ajratilganligi.
- Assistant javoblari bo‘sh emasligi.
- Unicode belgilarining to‘g‘ri saqlanganligi.
- Takroriy savol-javoblar mavjudligi.
- Noto‘g‘ri yoki manbasiz huquqiy javoblar mavjudligi.
- Train va validation o‘rtasida data leakage mavjudligi.

Har bir sample uchun manba modda va qonun raqamini imkon qadar saqla.

Noto‘g‘ri namunalarni avtomatik ravishda tuzatma. Ularni alohida review fayliga chiqar:

`data/reports/invalid_training_samples.jsonl`

Validation natijalarini:

`data/reports/dataset_validation_report.json`

fayliga yoz.

## 5. Dataset Format

Asosiy training formati chat messages bo‘lishi kerak.

Namuna:

    {
      "messages": [
        {
          "role": "user",
          "content": "Davlat boji nima?"
        },
        {
          "role": "assistant",
          "content": "Davlat boji — tegishli qonunchilikda belgilangan..."
        }
      ],
      "metadata": {
        "law_number": "O‘RQ-600",
        "article": "1-modda",
        "task_type": "legal_qa",
        "language": "uz"
      }
    }

Datasetdagi metadata modelga yuboriladigan chat matniga avtomatik qo‘shilmasligi kerak. Metadata faqat tracking va evaluation uchun ishlatiladi.

Agar model yoki TRL versiyasi boshqa dataset formatini talab qilsa, messages formatini shu talabga mos ravishda konvertatsiya qil.

## 6. Training Task Types

Datasetda quyidagi vazifalar mavjudligini tekshir:

### 6.1 Legal Question Answering

Foydalanuvchining huquqiy savoliga aniq javob berish.

### 6.2 Article Explanation

Qonun moddasini sodda va tushunarli tilda tushuntirish.

### 6.3 Article Retrieval

Savolga mos modda va bandni aniqlash.

### 6.4 Legal Text Summarization

Huquqiy matnning asosiy mazmunini qisqa bayon qilish.

### 6.5 Structured Extraction

Modda ichidagi qismlar, bandlar, sanalar va qiymatlarni ajratish.

### 6.6 Multi-Article Reasoning

Bir nechta moddadagi ma'lumotlarni birgalikda tahlil qilish.

### 6.7 Insufficient Evidence

Savolga javob berish uchun ma'lumot yetarli bo‘lmagan holatlarni aniqlash.

Agent barcha task turlarini tekshirishi, ammo manba bilan tasdiqlanmagan yangi savol-javoblarni training datasetga qo‘shmasligi kerak.

## 7. Environment Detection

Training boshlanishidan oldin Kaggle muhitini tekshir.

Aniqlanishi kerak:

- Python versiyasi.
- PyTorch versiyasi.
- CUDA mavjudligi.
- GPU modeli.
- GPU VRAM hajmi.
- CPU RAM.
- Diskdagi bo‘sh joy.
- Transformers versiyasi.
- TRL versiyasi.
- PEFT versiyasi.
- BitsAndBytes mavjudligi.

GPU mavjud bo‘lmasa yoki model uchun yetarli resurs bo‘lmasa, trainingni majburan boshlama.

`environment_report.json` yarat.

## 8. Base Model Selection

Gemma 4 modelining mos checkpointini tanla.

Model tanlashda:

- Gemma 4 ning rasmiy model checkpointidan foydalan.
- Instruction-tuned yoki IT checkpointini ustuvor tanla.
- Modelning litsenziya va foydalanish shartlarini tekshir.
- Hugging Face model sahifasidagi talablarni o‘rgan.
- Kaggle GPU xotirasiga mos model hajmini tanla.
- Transformers va boshqa kutubxonalarning model bilan mosligini tekshir.

Model nomini o‘zingdan taxmin qilib yaratma.

Agar kerakli model checkpointi Kaggle muhitida ishlamasa, sababini aniqlab, alternativ konfiguratsiya taklif qil.

## 9. Fine-Tuning Method

Boshlang‘ich usul sifatida LoRA yoki QLoRA dan foydalan.

### QLoRA

Agar GPU xotirasi cheklangan bo‘lsa, 4-bit quantization asosidagi QLoRA konfiguratsiyasini tanla.

Tavsiya etilgan boshlang‘ich parametrlar:

- Quantization: 4-bit
- Quantization type: NF4
- Double quantization: Enabled
- Compute dtype: GPU qo‘llab-quvvatlashiga qarab BF16 yoki FP16
- LoRA rank (r): 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Gradient checkpointing: Enabled

Bu parametrlar boshlang‘ich konfiguratsiya hisoblanadi. Ularni GPU xotirasi, model arxitekturasi va training natijalariga qarab o‘zgartir.

### LoRA Target Modules

Model arxitekturasini tekshir va LoRA qo‘llab-quvvatlaydigan layerlarni aniqlagin.

Target modules nomlarini taxmin qilib yozma.

Gemma 4 checkpointining haqiqiy modul nomlarini model strukturasi orqali tekshir.

Agar multimodal arxitektura mavjud bo‘lsa, faqat loyiha vazifasi uchun kerakli va qo‘llab-quvvatlanadigan qismlarni tanla.

## 10. Chat Template

Gemma 4 ning rasmiy chat template formatidan foydalan.

Training datasetdagi `messages` obyektlarini model tokenizerining chat template funksiyasi orqali qayta ishlagin.

Namuna:

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )

Lekin bu kodni modelning haqiqiy tokenizer API talablariga mosligini tekshirgandan keyingina ishlat.

Muhim qoidalar:

- Modelning maxsus tokenlarini qo‘lda taxmin qilib yozma.
- Chat template'ni o‘zingdan yaratma.
- Training va inference formatlari bir xil bo‘lishini ta'minla.
- User va assistant rollarini saqla.
- Assistant javoblarini trainingda to‘g‘ri maskalashni tekshir.

## 11. Training Configuration

TRL SFTTrainer yoki modelga mos rasmiy training usulidan foydalan.

Boshlang‘ich parametrlar:

- Learning rate: 2e-4 (LoRA uchun boshlang‘ich qiymat)
- Epochs: 2–3
- Per-device train batch size: 1–2
- Gradient accumulation steps: 4–8
- Warmup ratio: 0.03
- Weight decay: 0.01
- LR scheduler: cosine
- Logging steps: 10
- Save strategy: epoch yoki steps
- Evaluation strategy: epoch
- Seed: 42

Bu parametrlarni o‘zgarmas qiymat deb qabul qilma. Dataset hajmi, GPU xotirasi va model tavsiyalariga moslashtir.

### Sequence Length

Training sequence length modelning kontekst oynasi va GPU imkoniyatlariga qarab belgilanadi.

Uzun huquqiy savol-javoblar kesilib ketmasligini tekshir.

Agar sample belgilangan uzunlikdan oshsa:

- Muhim huquqiy matnni tasodifiy kesma.
- Datasetni to‘g‘ri chunklashni ko‘rib chiq.
- Truncation statistikasini hisobotda ko‘rsat.

## 12. Training Execution

Training jarayoni quyidagi ketma-ketlikda bajarilsin:

1. Modelni yuklash.
2. Tokenizerni yuklash.
3. Quantization konfiguratsiyasini qo‘llash.
4. LoRA adapterlarini sozlash.
5. Datasetni yuklash.
6. Chat template orqali formatlash.
7. Tokenization.
8. Training konfiguratsiyasini yaratish.
9. Validation dataset bilan baholash.
10. Fine-Tuning jarayonini boshlash.
11. Checkpointlarni saqlash.
12. Yakuniy model adapterini eksport qilish.

Training vaqtida quyidagilarni kuzat:

- Training loss.
- Validation loss.
- Learning rate.
- GPU memory usage.
- Training steps.
- Epoch progress.
- Training vaqti.

Agar CUDA out-of-memory xatosi yuz bersa, batch size yoki sequence length kabi parametrlarni ehtiyotkorlik bilan kamaytir.

## 13. Evaluation

Fine-Tuning tugagandan so‘ng modelni test qil.

Base model va Fine-Tuned model natijalarini bir xil test savollarida solishtir.

Quyidagi mezonlardan foydalan:

### Language Quality

- O‘zbek tilida ravon javob berishi.
- Huquqiy terminlarni to‘g‘ri ishlatishi.
- Grammatik xatolarni kamaytirishi.

### Legal Accuracy

- Modda raqamlarini to‘g‘ri ko‘rsatishi.
- Qonun matnini noto‘g‘ri talqin qilmasligi.
- Manbada yo‘q ma'lumotni yaratmasligi.
- Raqamli qiymatlarni o‘zgartirmasligi.

### Instruction Following

- Foydalanuvchi topshirig‘ini to‘g‘ri bajarishi.
- Kerakli formatda javob berishi.
- Yetarli ma'lumot bo‘lmaganda buni bildirishi.

### Metrics

Imkon qadar quyidagi ko‘rsatkichlarni hisobla:

- Validation Loss.
- Test Loss.
- Exact Match.
- Token-level F1.
- Answer Relevance.
- Citation Accuracy.
- Hallucination Rate.

Huquqiy javoblarni baholashda avtomatik metrikalar bilan birga manbaga asoslangan ekspert tekshiruvini ham qo‘llash kerak.

## 14. RAG Integration Test

Fine-Tuning tugagach, Gemma 4 modelini mavjud RAG tizimi bilan birlashtir.

Pipeline:

    User Question
          |
          v
    Query Processing
          |
          v
    FAISS + BM25 Retrieval
          |
          v
    Relevant Legal Context
          |
          v
    Fine-Tuned Gemma 4
          |
          v
    Grounded Legal Answer

Gemma 4 ga yuboriladigan system instruction:

"Sen O‘zbekiston Respublikasi qonunchiligi bo‘yicha yordamchi AI-assistentsan.

Foydalanuvchining savoliga faqat berilgan huquqiy kontekst va ishonchli manbalar asosida javob ber.

Har bir huquqiy xulosani tegishli qonun, modda va qism bilan asosla.

Agar kontekstda savolga javob berish uchun yetarli ma'lumot bo‘lmasa, buni ochiq bildir.

Qonun matnida mavjud bo‘lmagan raqam, sana, stavka yoki huquqiy normalarni o‘zingdan yaratma.

Tushuntirish va qonunning asl matnini bir-biridan aniq ajrat."

RAG testlarida retrieval natijalarini va model javoblarini alohida bahola.

## 15. Model Saving

Fine-Tuning natijalarini quyidagi strukturada saqla:

    models/
    ├── gemma4_legal_lora/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── tokenizer_files/
    │
    ├── checkpoints/
    │   └── checkpoint-*/
    │
    ├── evaluation/
    │   ├── base_model_results.json
    │   ├── finetuned_model_results.json
    │   └── rag_evaluation.json
    │
    └── reports/
        ├── training_report.json
        ├── environment_report.json
        └── model_card.md

Agar LoRA yoki QLoRA ishlatilgan bo‘lsa, avval adapterlarni alohida saqla.

Modelni merge qilish zarur bo‘lsa, faqat mos konfiguratsiya va qo‘llab-quvvatlanadigan usuldan foydalan.

Asl base modelni o‘zgartirma.

## 16. Reproducibility

Trainingni qayta bajarish imkoniyatini ta'minla.

Quyidagilarni saqla:

- Base model nomi va revision.
- Dataset versiyasi.
- Dataset hash.
- Tokenizer versiyasi.
- Training hyperparameters.
- LoRA configuration.
- Quantization configuration.
- Random seed.
- Kutubxonalar versiyasi.
- GPU modeli.
- Training sanasi.

Har bir training run uchun alohida konfiguratsiya fayli yarat.

## 17. Final Quality Report

Training yakunida `training_report.json` yarat.

Hisobotda quyidagilar bo‘lsin:

- Base model.
- Fine-Tuning usuli.
- Training samples soni.
- Validation samples soni.
- Test samples soni.
- Training epochs.
- Learning rate.
- LoRA rank.
- Quantization turi.
- Training loss.
- Validation loss.
- Evaluation natijalari.
- GPU xotirasi sarfi.
- Training davomiyligi.
- Saqlangan model manzili.
- Aniqlangan muammolar.

Base va Fine-Tuned model natijalarini taqqoslash jadvalini ham yarat.

## 18. Critical Rules

1. Asl huquqiy hujjatlarni o‘zgartirma.
2. Noto‘g‘ri dataset bilan training boshlama.
3. Base modelni asossiz almashtirma.
4. Model arxitekturasini tekshirmasdan LoRA target modules tanlama.
5. Chat template'ni tekshirmasdan training boshlama.
6. GPU resurslarini hisobga ol.
7. Training va validation datasetlarni aralashtirma.
8. Test datasetni trainingda ishlatma.
9. Overfitting holatini kuzat.
10. Model natijalarini manba asosida tekshir.
11. Fine-Tuning natijasi qonunning amaldagi tahririni avtomatik bilishini anglatmaydi.
12. Qonunchilikdagi o‘zgarishlar uchun RAG retrieval tizimidan foydalan.

## 19. Final Objective

Yakuniy maqsad — o‘zbek tilidagi huquqiy savollarni tushunadigan, huquqiy javoblarni tartibli shakllantiradigan va RAG orqali berilgan qonun kontekstiga tayangan holda ishlaydigan Gemma 4 modelini yaratish.

Fine-Tuning natijasi quyidagi arxitekturaga mos bo‘lishi kerak:

Fine-Tuned Gemma 4 + FAISS + BM25 + RAG

**Asosiy prinsip: Modelning javob berish qobiliyatini Fine-Tuning orqali yaxshila, qonunchilik faktlarini esa RAG orqali taqdim et.**

## 20. Existing Vector Index Integration

Vector Indexing bosqichi yakunlangan. Fine-Tuning agenti mavjud vector index fayllarini qaytadan yaratmasligi kerak.

### Existing Vector Index Structure

Loyihadagi mavjud papka strukturasi:

    vector_index/
    ├── bm25/
    │   └── bm25_index.pkl
    │
    ├── config/
    │   └── index_config.json
    │
    ├── embeddings/
    │   └── law_embeddings.npy
    │
    ├── faiss/
    │   └── law_index.faiss
    │
    ├── metadata/
    │   ├── chunk_mapping.json
    │   └── chunks.json
    │
    └── reports/
        ├── indexing_report.json
        └── retrieval_evaluation.json

### 20.1. Index Files Validation

Fine-Tuning boshlanishidan oldin quyidagi fayllar mavjudligini tekshir:

- `vector_index/bm25/bm25_index.pkl`
- `vector_index/config/index_config.json`
- `vector_index/embeddings/law_embeddings.npy`
- `vector_index/faiss/law_index.faiss`
- `vector_index/metadata/chunk_mapping.json`
- `vector_index/metadata/chunks.json`
- `vector_index/reports/indexing_report.json`
- `vector_index/reports/retrieval_evaluation.json`

Barcha fayllarning o‘qilishini tekshir.

FAISS index, embeddinglar va metadata o‘rtasidagi moslikni tekshir.

Agar muammo aniqlansa, mavjud indexni o‘zgartirma. Muammo haqida hisobot yarat.

### 20.2. Important Separation

Vector index fayllarini Fine-Tuning dataset sifatida ishlatma.

Quyidagi fayllar SFT uchun asosiy training input hisoblanadi:

- `data/dataset/train.jsonl`
- `data/dataset/validation.jsonl`
- `data/dataset/test.jsonl`

Vector index esa Fine-Tuning yakunlangandan so‘ng RAG integratsiyasi va retrieval testlarida ishlatiladi.

### 20.3. RAG Integration

Fine-Tuning yakunlangach mavjud indexlarni yukla:

FAISS:
`vector_index/faiss/law_index.faiss`

BM25:
`vector_index/bm25/bm25_index.pkl`

Chunks:
`vector_index/metadata/chunks.json`

Chunk mapping:
`vector_index/metadata/chunk_mapping.json`

Configuration:
`vector_index/config/index_config.json`

Foydalanuvchi savoli kelganda:

1. Query embedding yarat.
2. FAISS orqali semantik qidiruv bajar.
3. BM25 orqali lexical qidiruv bajar.
4. Natijalarni RRF yordamida birlashtir.
5. Tegishli chunklarni metadata bilan birga ol.
6. Ularni Fine-Tuned Gemma 4 modeliga context sifatida yubor.
7. Javobda qonun nomi va modda raqamini ko‘rsat.

### 20.4. Final Integration Test

Fine-Tuned modelni mavjud vector index bilan birgalikda sinovdan o‘tkaz.

Quyidagi testlarni bajar:

- Oddiy huquqiy savol.
- Aniq modda bo‘yicha savol.
- Moddaning muayyan qismi bo‘yicha savol.
- Jadvaldagi davlat boji stavkasi bo‘yicha savol.
- Bir nechta moddani talab qiladigan savol.
- Indexda javobi topilmaydigan savol.

Har bir testda retrieval natijasi va Gemma 4 javobini alohida bahola.

### 20.5. Index Preservation

Fine-Tuning davomida:

- `vector_index/` papkasini o‘chirma.
- FAISS indexni qayta yaratma.
- BM25 indexni qayta yaratma.
- Embeddinglarni qayta hisoblama.
- Chunk ID va metadata mosligini buzma.

Faqat alohida talab yoki xatolik aniqlanganda yangi index yaratish jarayonini boshlash mumkin.