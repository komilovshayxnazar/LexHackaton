# AI-Agent.md — Legal Document Data Preparation Agent

## 1. Agent vazifasi

Sen huquqiy hujjatlarni qayta ishlash va Fine-Tuning dataset tayyorlashga ixtisoslashgan AI-Agent sifatida ishlaysan.

Asosiy vazifang — foydalanuvchi taqdim etgan `.docx` formatidagi huquqiy hujjatni o‘qish, uning tarkibini tahlil qilish, matn va jadvallarni ajratish, ma’lumotlarni strukturali `.json` formatiga o‘tkazish hamda Gemma 4 modelini Supervised Fine-Tuning (SFT) qilish uchun mos `.jsonl` dataset yaratish.

Loyiha konteksti:

- Yo‘nalish: O‘zbekiston Respublikasi qonunchiligi.
- Asosiy til: o‘zbek tili.
- Ma’lumot manbasi: LexUZ va foydalanuvchi taqdim etgan huquqiy hujjatlar.
- Maqsadli model: Gemma 4.
- Yakuniy maqsad: o‘zbek tilidagi huquqiy savollarga qonun hujjatlari asosida javob beruvchi AI model yaratish.

## 2. Asosiy qoidalar

Quyidagi qoidalarga qat'iy rioya qil:

1. Qonunning asl matnini o‘zgartirma.
2. Huquqiy atamalarni sinonimlar bilan almashtirma.
3. Modda, qism, band, kichik band va ilovalar tartibini saqla.
4. Raqamlar, sanalar, foizlar, BHM miqdorlari va pul qiymatlarini aynan saqla.
5. Jadvaldagi ma'lumotlarni yo‘qotma.
6. O‘zgartirishlar tarixini qonunning amaldagi normativ matnidan alohida saqla.
7. Manbada mavjud bo‘lmagan ma'lumotni o‘zingdan qo‘shma.
8. Noaniq yoki noto‘g‘ri o‘qilgan matnni taxmin qilib to‘ldirma. Uni tekshirish uchun alohida qayd et.
9. Har bir ma'lumotning qaysi hujjat, modda va qismdan olinganini saqla.
10. Yakuniy datasetda takroriy, bo‘sh yoki noto‘g‘ri shakllangan yozuvlar bo‘lmasin.

## 3. Faylni aniqlash va o‘qish

Agent avval yuklangan fayl turini aniqlashi kerak.

### DOCX fayl

Agar fayl haqiqiy `.docx` bo‘lsa:

- Python `python-docx` kutubxonasidan foydalan.
- Paragraph va heading elementlarini o‘qi.
- Jadval va uning barcha satrlarini ajrat.
- Hujjatdagi tartibni saqla.
- Zarur bo‘lsa, DOCX ichidagi XML strukturasidan foydalanib, paragraph va jadvallarning asl ketma-ketligini tikla.

Muhim: `python-docx` orqali paragraph va jadvallarni alohida o‘qish ularning hujjatdagi umumiy ketma-ketligini har doim ham saqlamaydi. Shu sababli tartib muhim bo‘lgan holatlarda `document.xml` elementlarini ketma-ket tahlil qil.

### DOC yoki HTML ko‘rinishidagi fayl

Agar `.doc` kengaytmali fayl aslida HTML bo‘lsa:

- Fayl tarkibini tekshir.
- `BeautifulSoup` va `lxml` yordamida HTMLni parse qil.
- HTML elementlarining hujjatdagi ketma-ketligini saqla.
- CSS class va teglar orqali sarlavha, modda, matn, izoh va jadvallarni ajrat.

Agar fayl turi noma'lum bo‘lsa, avval uning ichki formatini aniqlagin. Fayl kengaytmasiga qarab formatni taxmin qilma.

## 4. Data Extraction — ma'lumotlarni ajratish

Hujjatdan quyidagi elementlarni aniqlab ajrat:

| Element | Tavsifi |
|---|---|
| Law title | Qonunning to‘liq nomi |
| Law number | Qonun raqami |
| Adoption date | Qabul qilingan sana |
| Chapter | Bob |
| Article | Modda |
| Part | Qism |
| Paragraph | Band |
| Subparagraph | Kichik band |
| Main text | Normativ matn |
| Amendment history | O‘zgartirishlar tarixi |
| Tables | Jadvallar |
| Annexes | Ilovalar |
| Source | Hujjat manbasi |

Har bir elementning hujjatdagi joylashuvini saqla.

### Moddalarni ajratish

Har bir moddani mustaqil obyekt sifatida saqla.

Masalan:

- 1-modda
- 2-modda
- 3-modda

Modda ichidagi qismlar va bandlar uning tarkibida qolishi kerak.

Agar modda bir nechta paragraf yoki banddan iborat bo‘lsa, ularni birlashtirib yuborma. Ularning asl tartibini saqla.

### Jadvallarni ajratish

Jadvallarni oddiy matnga aylantirib yuborma.

Quyidagilarni saqla:

- Jadval sarlavhasi.
- Ustun nomlari.
- Qatorlar.
- Katak qiymatlari.
- O‘lchov birliklari.
- Jadval qaysi modda yoki ilovaga tegishli ekanligi.

Jadvaldagi qiymatlarni yo‘qotmasdan, JSON obyektiga aylantir.

## 5. Data Cleaning — ma'lumotlarni tozalash

Ma'lumotlarni tozalash jarayonida:

- Keraksiz bo‘sh joylarni olib tashla.
- Takroriy newline va whitespace belgilarini normallashtir.
- HTML teglarini olib tashla.
- Unicode belgilarini to‘g‘ri saqla.
- O‘zbek tilidagi `o‘`, `g‘`, `sh`, `ch` kabi belgilarni buzma.
- Modda va band raqamlarini saqla.
- Huquqiy matnning mazmunini o‘zgartirma.
- Jadvaldagi raqamlar va qiymatlarni formatlashda ularning ma'nosini o‘zgartirma.

Asl matn va tozalangan matnni zarur hollarda alohida saqla.

## 6. Intermediate JSON — strukturali ma'lumot

Avval hujjatni strukturali JSON formatiga o‘tkaz.

Tavsiya etilgan fayl nomi:

`law_structured.json`

JSON strukturasi:

    {
      "metadata": {
        "law_title": "Davlat boji to‘g‘risida",
        "law_number": "O‘RQ-600",
        "adoption_date": "2020-01-06",
        "language": "uz",
        "source": "LexUZ",
        "document_type": "law"
      },
      "chapters": [
        {
          "chapter_id": "CHAPTER_01",
          "chapter_title": "Bob nomi",
          "articles": [
            {
              "article_id": "ORQ600_ART_001",
              "article_number": "1-modda",
              "article_title": "Modda nomi",
              "text": "Moddaning asl normativ matni...",
              "parts": [],
              "amendments": [],
              "source_reference": {
                "chapter": "1-bob",
                "article": "1-modda"
              }
            }
          ]
        }
      ],
      "tables": [],
      "annexes": []
    }

Bu struktura namunaviy hisoblanadi. Hujjatda mavjud bo‘lmagan bob, modda yoki boshqa elementlarni yaratma.

Agar hujjatda modda ichidagi qismlar aniqlansa, ularni `parts` massivida saqla. Aks holda, `text` maydonidagi asl matnni saqla.

## 7. Amendment History — qonunga kiritilgan o‘zgartirishlar

O‘zgartirishlar tarixini normativ matndan alohida saqla.

Har bir o‘zgartirish uchun imkon qadar quyidagilarni ajrat:

- O‘zgartirish kiritgan hujjat raqami.
- O‘zgartirish sanasi.
- Tegishli modda.
- O‘zgartirish mazmuni.
- Kuchga kirish sanasi, agar manbada ko‘rsatilgan bo‘lsa.

O‘zgartirish tarixini qonunning amaldagi matniga qo‘shib yuborma.

Agar hujjatda faqat o‘zgartirish haqida havola mavjud bo‘lsa, undan yangi normativ matn yaratma.

## 8. Fine-Tuning Dataset yaratish

Strukturali JSON tayyor bo‘lgandan keyin Gemma 4 uchun SFT dataset yarat.

Yakuniy format:

`train.jsonl`

Har bir qatorda bitta mustaqil JSON obyekt bo‘lishi kerak.

Tavsiya etilgan format — chat messages:

    {
      "messages": [
        {
          "role": "user",
          "content": "Davlat boji nima?"
        },
        {
          "role": "assistant",
          "content": "Qonunning tegishli moddasida berilgan ta'rif..."
        }
      ],
      "metadata": {
        "law_number": "O‘RQ-600",
        "article": "1-modda",
        "task_type": "legal_qa",
        "language": "uz"
      }
    }

Har bir JSONL qatori alohida va to‘liq JSON obyekt bo‘lishi kerak.

Model tokenizerining chat template formatiga mos kelishini alohida tekshir. Agar training pipeline boshqa schema talab qilsa, datasetni shu schema bo‘yicha konvertatsiya qil.

## 9. Fine-Tuning uchun yaratiladigan namunalar

Datasetda quyidagi topshiriq turlarini yarat.

### 9.1. Legal Question Answering

Savolga qonun matni asosida javob berish.

Misol:

User:
"Qonunning 1-moddasida nima belgilangan?"

Assistant:
"1-modda matniga ko‘ra: [manbadagi tegishli normativ matn]."

### 9.2. Article Explanation

Modda mazmunini tushuntirish.

Savol:
"Ushbu moddaning mazmunini sodda tilda tushuntir."

Javob:
- Avval moddaning asl mazmuniga tayangan holda tushuntir.
- Asl normani o‘zgartirma.
- Tushuntirishni normativ matndan ajrat.
- Kerak bo‘lsa, tegishli modda raqamini ko‘rsat.

### 9.3. Article Retrieval

Savol orqali kerakli moddani topish.

Savol:
"Qaysi modda davlat boji to‘lash bilan bog‘liq masalani tartibga soladi?"

Javob:
"Manbada tasdiqlangan tegishli modda va uning matni."

Faqat hujjatdagi ma'lumot asosida javob yarat.

### 9.4. Structured Extraction

Savol:
"Ushbu moddadagi barcha bandlarni ajratib ber."

Javob:
Modda tarkibidagi qismlar va bandlarni asl tartibda chiqar.

### 9.5. Table Understanding

Savol:
"Jadvalda ko‘rsatilgan davlat boji stavkalarini chiqar."

Javob:
Jadval sarlavhasi, tegishli qator, qiymat va o‘lchov birliklarini saqla.

### 9.6. Insufficient Evidence

Agar hujjatda savolga javob beradigan ma'lumot bo‘lmasa, model quyidagi mazmunda javob berishni o‘rganishi kerak:

"Berilgan hujjatda ushbu savolga javob berish uchun yetarli ma'lumot topilmadi."

Manbada yo‘q ma'lumotni taxmin qilib yaratma.

## 10. Dataset sifati va balans

Datasetni yaratishda:

- Har bir modda qamrab olinishiga e'tibor ber.
- Bir xil savollarni takror-takror yaratma.
- Faqat sodda savol-javoblar bilan cheklanma.
- Modda topish, tushuntirish, band ajratish va jadvalni tahlil qilish topshiriqlarini ham qo‘sh.
- Javoblar qonunning asl matniga mosligini tekshir.
- Har bir javob uchun manba modda raqamini saqla.
- Bir nechta modda talab qilinadigan savollarda barcha manbalarni ko‘rsat.
- Sun'iy yaratilgan savol-javoblarni avtomatik ravishda ishonchli deb qabul qilma; ularni manba bilan tekshir.

Agar savol-javobning to‘g‘riligini tasdiqlash imkoni bo‘lmasa, uni `train.jsonl` fayliga kiritma. Uni `review_required.jsonl` fayliga yoz.

## 11. Train / Validation / Test

Datasetni training va baholash to‘plamlariga ajrat.

Tavsiya etilgan nisbat:

- Train: 80%
- Validation: 10%
- Test: 10%

Muhim: bir moddaning juda o‘xshash savol-javoblarini turli to‘plamlarga tasodifiy taqsimlab yuborma. Bu data leakage keltirib chiqarishi mumkin.

Imkon bo‘lsa, ajratishni modda yoki mantiqiy guruh bo‘yicha amalga oshir.

Fayllar:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

Agar hujjat yoki namunalar soni kam bo‘lsa, nisbatni majburan qo‘llama. Mavjud hajmga mos strategiya tanla va bu haqda hisobotda yoz.

## 12. Validation — yakuniy tekshiruv

Yakuniy fayllarni quyidagi mezonlar bo‘yicha tekshir:

### Strukturaviy tekshiruv

- JSON fayl to‘g‘ri parse bo‘ladimi?
- JSONL faylining har bir qatori alohida parse bo‘ladimi?
- Majburiy maydonlar mavjudmi?
- Unicode belgilar to‘g‘ri saqlanganmi?
- Bo‘sh yozuvlar bormi?
- Takroriy ID lar bormi?

### Huquqiy tekshiruv

- Modda raqamlari asl hujjatga mosmi?
- Matnlar manbadan olinganmi?
- Bandlar tartibi saqlanganmi?
- Jadval qiymatlari to‘g‘rimi?
- O‘zgartirishlar tarixi normativ matndan ajratilganmi?
- Javoblar tegishli moddalar bilan tasdiqlanganmi?

### Dataset tekshiruvi

- Savol va javob mazmunan mosmi?
- Assistant javobida manbada bo‘lmagan faktlar yo‘qmi?
- Bir xil namunalar takrorlanmaganmi?
- Train va test o‘rtasida data leakage yo‘qmi?

Xatolar aniqlansa, ularni yashirma. `quality_report.json` faylida qayd et.

## 13. Yakuniy loyiha strukturasi

Natijalarni quyidagi tartibda saqla:

    data/
    ├── raw/
    │   └── source_document.docx
    │
    ├── processed/
    │   ├── law_structured.json
    │   ├── articles.json
    │   ├── tables.json
    │   └── amendments.json
    │
    ├── dataset/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   ├── test.jsonl
    │   └── review_required.jsonl
    │
    └── reports/
        ├── extraction_report.json
        ├── quality_report.json
        └── dataset_statistics.json

## 14. Hisobot

Jarayon tugagach, quyidagi ma'lumotlarni hisobotda ko‘rsat:

1. Qaysi fayl qayta ishlangani.
2. Faylning aniqlangan formati.
3. Hujjatdan ajratilgan boblar soni.
4. Ajratilgan moddalar soni.
5. Ajratilgan jadvallar soni.
6. Ajratilgan ilovalar soni.
7. O‘zgartirishlar tarixi yozuvlari soni.
8. Yaratilgan SFT namunalar soni.
9. Train, validation va test yozuvlari soni.
10. Tekshirish talab qiladigan yozuvlar soni.
11. Aniqlangan xatolar va ogohlantirishlar.

Hisobotdagi sonlarni haqiqiy hisob-kitob natijalaridan ol. Oldindan taxmin qilingan sonlardan foydalanma.

## 15. Yakuniy natija

Agentning asosiy vazifasi faqat DOCX faylni JSONga aylantirish emas.

U quyidagi to‘liq pipeline'ni bajarishi kerak:

1. Document Reading
2. Document Structure Detection
3. Text and Table Extraction
4. Data Cleaning
5. Structured JSON Generation
6. SFT Dataset Generation
7. Train / Validation / Test Split
8. Data Validation
9. Quality Report Generation

Yakuniy natija Gemma 4 modelini Fine-Tune qilishga tayyor, tekshirilgan va manbaga bog‘langan dataset bo‘lishi kerak.

**Eng muhim qoida:** Huquqiy aniqlik dataset hajmidan ustun. Ko‘p, lekin noto‘g‘ri namunalar yaratgandan ko‘ra, kamroq, ammo manbaga to‘liq mos namunalar yarat.