# DANA Review Insights

Analisis ulasan Google Play aplikasi DANA (31 Okt – 17 Nov 2026, 11.136 ulasan bersih): topik, sentimen, ringkasan.

## Struktur repo

```
dana-review-insights/
├── 01_scraper.ipynb                              [Orang 1] scraping, cleaning, EDA
├── 02_nlp_topik_sentimen_ringkasan.ipynb      [Orang 2] LDA, sentimen, ringkasan, evaluasi
├── 03_analisis_visualisasi.ipynb              [Orang 3] grafik dan tabel insight (siap dijalankan)
├── requirements.txt
├── data/
│   ├── raw/reviews_raw.csv                    [Orang 1]
│   ├── clean/reviews_clean.csv                [Orang 1] input NLP
│   ├── labels/label_manual.csv                [Orang 1] 100 ulasan berlabel manual
│   └── processed/reviews_topics_sentiment.csv [Orang 2] HASIL NLP, input Orang 3
└── reports/
    ├── eda_overview.png, eda_per_hari.png     [Orang 1]
    ├── coherence_k.png                        [Orang 2] pemilihan k LDA
    ├── ringkasan_per_topik.md                 [Orang 2] ringkasan per topik
    ├── evaluasi_nlp.md                        [Orang 2] hasil evaluasi vs label manual
    ├── tabel_insight.csv, figures/*.png       [Orang 3] keluaran notebook 03
```

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook          # jalankan dari folder utama repo
```

- **Orang 3:** buka `03_analisis_visualisasi.ipynb` lalu **Run all**. Cukup membaca `data/processed/reviews_topics_sentiment.csv`
  (tidak ada model yang dilatih). Di Google Colab: clone repo, `%cd dana-review-insights`, lalu jalankan.
- **Mengulang tahap NLP:** buka `02_nlp_topik_sentimen_ringkasan.ipynb` (disarankan Colab + GPU T4 untuk IndoBERT). Hasilnya sama karena `random_state` dikunci.

## Kolom hasil NLP (`reviews_topics_sentiment.csv`)

| Kolom | Arti |
|---|---|
| `topic_id`, `topic_label`, `topic_prob` | topik dominan LDA (k = 8) dan keyakinannya |
| `low_conf` | `True` = topik kurang yakin (±28% data, umumnya ulasan sangat pendek) |
| `sentimen` | **sentimen final** (IndoBERT/RoBERTa): positif / netral / negatif. **Pakai kolom ini.** |
| `sentimen_bert`, `sentimen_tfidf` | hasil tiap model. `sentimen_tfidf` hanya baseline pembanding |
| `versi_grp` | versi aplikasi yang dikelompokkan |
| `bertopic_id`, `pseudo`, `sentimen_bert_skor` | kolom pendukung, tidak dipakai di analisis utama |

## Hasil evaluasi (n = 100 ulasan berlabel manual)

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
