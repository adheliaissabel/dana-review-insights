# DANA Review Insights
 
Analisis ulasan Google Play aplikasi DANA (31 Okt – 17 Nov 2026, 11.136 ulasan bersih): topik, sentimen, ringkasan.
 
Proyek dibagi dalam tiga tahap sesuai penomoran file:
 
| No. | Tahap | Berkas utama |
|---|---|---|
| **01** | Pengumpulan, pembersihan, dan EDA data | `01_scraper.ipynb` |
| **02** | NLP: topic modeling, sentimen, ringkasan, evaluasi | `02_nlp_topik_sentimen_ringkasan.ipynb` |
| **03** | Analisis dan visualisasi untuk wawasan produk | `03_analisis_visualisasi.ipynb` (menyusul) |
 
## Struktur repo
 
```
dana-review-insights/
├── 01_scraper.ipynb                           [01] scraping, cleaning, EDA
├── 02_nlp_topik_sentimen_ringkasan.ipynb      [02] LDA, sentimen, ringkasan, evaluasi
├── 03_analisis_visualisasi.ipynb              [03] (menyusul)
├── requirements.txt
├── data/
│   ├── raw/reviews_raw.csv                    [01] data mentah
│   ├── clean/reviews_clean.csv                [01] data bersih, input tahap 02
│   ├── labels/label_manual.csv                [01] 100 ulasan berlabel manual (evaluasi 02)
│   └── processed/reviews_topics_sentiment.csv [02] hasil NLP, input tahap 03
└── reports/
    ├── eda_overview.png, eda_per_hari.png     [01]
    ├── coherence_k.png                        [02] pemilihan k LDA
    ├── ringkasan_per_topik.md                 [02] ringkasan per topik
    └── evaluasi_nlp.md                        [02] evaluasi terhadap label manual
```
 
Arah data: `raw → clean → processed → (03)`.
 
## Cara menjalankan
 
```bash
pip install -r requirements.txt
jupyter notebook          # jalankan dari folder utama repo
```
 
- **01:** `01_scraper.ipynb` (scraping membutuhkan internet).
- **02:** `02_nlp_topik_sentimen_ringkasan.ipynb`. Disarankan Google Colab dengan GPU T4 untuk IndoBERT. Hasilnya sama karena `random_state` dikunci.
- **03:** membaca `data/processed/reviews_topics_sentiment.csv`; tidak ada model yang dilatih.
## Kolom hasil NLP (`reviews_topics_sentiment.csv`)
 
| Kolom | Arti |
|---|---|
| `topic_id`, `topic_label`, `topic_prob` | topik dominan LDA (k = 8) dan keyakinannya |
| `low_conf` | `True` = topik kurang yakin (±28% data, umumnya ulasan sangat pendek) |
| `sentimen` | **sentimen final** (IndoBERT/RoBERTa): positif / netral / negatif |
| `sentimen_bert`, `sentimen_tfidf` | hasil tiap model; `sentimen_tfidf` hanya baseline pembanding |
| `versi_grp` | versi aplikasi yang dikelompokkan |
| `bertopic_id`, `pseudo`, `sentimen_bert_skor` | kolom pendukung |
 
## Hasil evaluasi tahap 02 (n = 100 ulasan berlabel manual)
 
| | Hasil |
|---|---|
| Topik cocok | 49,0% (54,4% untuk ulasan yang yakin) |
| Sentimen IndoBERT/RoBERTa | accuracy 0,72, macro-F1 0,678 |
| Sentimen baseline TF-IDF | accuracy 0,67, macro-F1 0,647 |
 
Detail dan tabel silang di `reports/evaluasi_nlp.md`.
 
## Keterbatasan
 
- Topik adalah perkiraan; label ditafsirkan manusia dari kata teratas LDA. Dua topik event dan dua topik transaksi cukup mirip.
- Kelas netral lemah, sehingga porsi sentimen negatif cenderung terlalu tinggi.
- Lonjakan 6 Nov (±33% data) memengaruhi topik dan perbandingan versi; selalu bandingkan juga tanpa `is_spike`.
- Rentang data hanya ±17 hari; temuan "bug versi terbaru" bersifat indikasi.
- Evaluasi memakai 100 ulasan, jadi selisih kecil antar model belum tentu nyata.