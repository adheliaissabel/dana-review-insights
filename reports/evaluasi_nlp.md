# Evaluasi NLP terhadap label manual

Sampel: **100 ulasan** berlabel manual oleh Orang 1 (35 hari lonjakan 6 Nov + 65 hari biasa). Sampel kecil: selisih beberapa poin persen belum tentu nyata.

## 1. Topic modeling (LDA, k = 8)

- Coherence c_v tertinggi pada semua data: k = 8 (0,452). Lihat `coherence_k.png`.
- Topik cocok dengan label manual (setelah label LDA dipetakan ke label Orang 1): **49.0%** (n=100).
- Hanya ulasan yang yakin (bukan `low_conf`): **54.4%** (n=57).
- Cocok terbaik: Pujian Umum (12 dari 17). Gangguan Jaringan 16 dari 32. Login/Akun sering terbaca sebagai Saldo/Transfer (7 dari 8) karena kata "masuk" ambigu.
- `ui_fitur` (2 ulasan) dan `lainnya` (8) tidak punya topik LDA tersendiri.

Tabel silang (baris = label manual, kolom = topik LDA yang dipetakan):

| topik_manual      |   cicilan_limit |   event_promo |   gangguan_jaringan |   lainnya |   login_akun |   pujian_umum |   transaksi_saldo |
|:------------------|----------------:|--------------:|--------------------:|----------:|-------------:|--------------:|------------------:|
| cicilan_limit     |               3 |             2 |                   0 |         0 |            0 |             1 |                 0 |
| event_promo       |               1 |             5 |                   0 |         0 |            1 |             1 |                 0 |
| gangguan_jaringan |               1 |             3 |                  16 |         0 |            4 |             2 |                 6 |
| lainnya           |               1 |             1 |                   0 |         2 |            1 |             2 |                 1 |
| login_akun        |               1 |             0 |                   0 |         0 |            0 |             0 |                 7 |
| pujian_umum       |               0 |             1 |                   3 |         0 |            0 |            12 |                 1 |
| transaksi_saldo   |               2 |             1 |                   2 |         0 |            0 |             3 |                11 |
| ui_fitur          |               0 |             0 |                   0 |         0 |            1 |             0 |                 1 |


## 2. Sentimen (3 kelas) terhadap label manual

|                |   accuracy |   macro_F1 |
|:---------------|-----------:|-----------:|
| sentimen_tfidf |       0.67 |      0.647 |
| sentimen_bert  |       0.72 |      0.678 |

Model final: **IndoBERT/RoBERTa (`sentimen_bert`)** karena accuracy dan macro-F1 lebih tinggi. Selisihnya kecil (n=100), jadi baseline TF-IDF disimpan sebagai pembanding.

### sentimen_bert

```
              precision    recall  f1-score   support

     negatif      0.667     0.958     0.786        48
      netral      0.636     0.233     0.341        30
     positif      0.950     0.864     0.905        22

    accuracy                          0.720       100
   macro avg      0.751     0.685     0.678       100
weighted avg      0.720     0.720     0.679       100

```
Confusion matrix (baris = manual, kolom = prediksi; urutan negatif, netral, positif):

```
[[46  1  1]
 [23  7  0]
 [ 0  3 19]]
```

### sentimen_tfidf

```
              precision    recall  f1-score   support

     negatif      0.723     0.708     0.716        48
      netral      0.647     0.367     0.468        30
     positif      0.611     1.000     0.759        22

    accuracy                          0.670       100
   macro avg      0.661     0.692     0.647       100
weighted avg      0.676     0.670     0.651       100

```
Confusion matrix (baris = manual, kolom = prediksi; urutan negatif, netral, positif):

```
[[34  6  8]
 [13 11  6]
 [ 0  0 22]]
```

Catatan: kelas netral lemah di kedua model (IndoBERT: 23 dari 30 ulasan netral terbaca negatif), sehingga porsi negatif cenderung terlalu tinggi.

## 3. Ringkasan

Evaluasi kualitatif ringkasan (kejelasan dan akurasi isi) belum diisi; lihat tabel penilaian di notebook 02.