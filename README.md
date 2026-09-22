# Predictive Maintenance — Klasifikasi Tipe Kegagalan Mesin

Proyek ini membangun model klasifikasi multiclass untuk memprediksi **tipe kegagalan mesin** (`Failure Type`), dengan tujuan bisnis utama: **beralih dari reactive maintenance ke predictive maintenance** guna mengurangi biaya downtime produksi akibat kerusakan tak terduga dan mengoptimalkan jadwal maintenance preventif secara lebih efisien.

## Latar Belakang Bisnis

Bagi perusahaan yang mengoperasikan peralatan/mesin produksi, kegagalan mendadak menimbulkan dua jenis kerugian sekaligus: **biaya downtime** yang menghentikan produksi, dan **biaya maintenance reaktif** yang lebih mahal dibanding perbaikan terjadwal. Model prediktif ini membantu tim maintenance:

- Mengetahui **jenis kegagalan spesifik** yang berpotensi terjadi (bukan sekadar "akan gagal/tidak"), sehingga tindakan preventif bisa lebih terarah (mis. mengganti tool vs mendinginkan mesin vs memeriksa beban torsi)
- Mengalokasikan sumber daya maintenance berdasarkan **fitur operasional yang paling berisiko**, bukan jadwal kaku berbasis waktu
- Memahami **trade-off nyata** antara mengejar akurasi keseluruhan vs kemampuan mendeteksi kegagalan langka namun kritis

## Dataset

| Detail | Keterangan |
|---|---|
| Target | `Failure Type` (multiclass): No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, Random Failures |
| Target biner pendukung | `Target` (0 = No Failure, 1 = Failure) |
| Fitur sensor | Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] |
| Fitur kategorikal | `Type` (L/M/H — tipe/kapasitas mesin) |
| Karakteristik data | **Sangat imbalanced** — kelas `No Failure` mendominasi, sementara `Random Failures` dan `Tool Wear Failure` adalah kelas minoritas ekstrem |

## Metodologi

1. **EDA** — Memeriksa distribusi target, korelasi fitur sensor terhadap kegagalan, dan distribusi tipe mesin.
2. **Data Cleaning** — Menghapus kolom ID non-informatif (`UDI`, `Product ID`), One-Hot Encoding untuk `Type`, serta pengecekan missing value dan duplikasi (tidak ditemukan).
3. **Feature Engineering** — Membuat dua fitur turunan berbasis domain teknik:
   - `Power` = Torque × Rotational speed (representasi beban kerja mekanis mesin)
   - `Temp_Diff` = Process temperature − Air temperature (indikator stres termal/efisiensi pendinginan)
4. **Split & Handling Imbalance** — Split 80:20 dengan stratifikasi, dilanjutkan dua strategi penanganan data imbalanced:
   - **Class weighting** (`class_weight='balanced'`) — memberi bobot lebih besar pada kelas minoritas saat training, tanpa mengubah data
   - **SMOTE** (Synthetic Minority Over-sampling) — membuat sampel sintetis kelas minoritas agar seimbang dengan kelas mayoritas di data training
5. **Modeling** — Melatih dan membandingkan 4 kombinasi model:
   - Logistic Regression + class_weight
   - Logistic Regression + SMOTE
   - Random Forest + class_weight
   - Random Forest + SMOTE
6. **Feature Importance** — Mengekstraksi bobot kontribusi tiap fitur dari model Random Forest untuk validasi domain knowledge.

## Hasil Perbandingan Model

| Model | Accuracy | Macro F1 | Recall: Heat Dissipation | Recall: Overstrain | Recall: Power | Recall: Tool Wear | Recall: Random Failures |
|---|---|---|---|---|---|---|---|
| Logistic Regression + class_weight | 0.96 | 0.48 | 0.86 | 1.00 | 0.79 | **0.00** | 0.00 |
| Logistic Regression + SMOTE | 0.67 | 0.32 | 0.91 | 0.94 | 0.95 | **0.89** | 0.00 |
| **Random Forest + class_weight** | **0.97** | **0.64** | 1.00 | 0.75 | 1.00 | **0.22** | 0.00 |
| Random Forest + SMOTE | 0.91 | 0.62 | ~0.91 | tinggi | tinggi | 0.22 | 0.00 |

> Catatan: `Random Failures` gagal terdeteksi (Recall 0.00) oleh **seluruh** model yang diuji, tanpa terkecuali.

## Feature Importance (Random Forest)

| Peringkat | Fitur | Importance |
|---|---|---|
| 1 | `Tool wear [min]` | 22.7% |
| 2 | `Power` (fitur turunan) | 20.6% |
| 3 | `Temp_Diff` (fitur turunan) | 16.3% |
| 4 | `Torque [Nm]` | 15.1% |
| 5 | `Rotational speed [rpm]` | 12.4% |
| 6 | `Process temperature [K]` | 5.5% |
| 7 | `Air temperature [K]` | 5.2% |
| 8–10 | `Type` (L/M/H) | < 1% masing-masing |

**Insight penting:** Kedua fitur hasil rekayasa (`Power` dan `Temp_Diff`) menempati peringkat 2 dan 3, mengonfirmasi bahwa kombinasi variabel operasional (bukan sekadar sensor mentah) memberikan sinyal yang lebih kuat terhadap risiko kegagalan mesin — memvalidasi pendekatan feature engineering berbasis domain teknik.

## Analisis Trade-off: Mengapa Tidak Ada "Model Terbaik" Tunggal

Fenomena paling menarik dari proyek ini: **`Tool wear [min]` adalah fitur importance tertinggi (22.7%), namun Random Forest — model dengan performa keseluruhan terbaik — justru paling lemah dalam mendeteksi `Tool Wear Failure` (Recall hanya 0.22)**. Sebaliknya, Logistic Regression + SMOTE mencapai Recall 0.89 untuk kelas yang sama, meski akurasi keseluruhannya jauh lebih rendah.

Ini menunjukkan kaidah penting dalam interpretasi model pada data imbalanced:

> **Feature importance yang tinggi tidak menjamin recall yang tinggi pada kelas minor.** Feature importance mengukur kontribusi fitur terhadap keputusan model secara agregat (yang didominasi kelas mayoritas), bukan kemampuan spesifik model memisahkan kelas minoritas tersebut. Model tree-based seperti Random Forest cenderung tetap konservatif terhadap kelas langka meski fitur diskriminatifnya kuat, sementara teknik oversampling seperti SMOTE secara eksplisit memaksa model mempelajari pola minoritas — dengan konsekuensi presisi menurun karena batas keputusan menjadi lebih longgar.

Sehingga **pertimbangan pemilihan model tidak bisa hanya bersandar pada satu metrik** (akurasi atau feature importance saja), melainkan harus disesuaikan dengan **prioritas bisnis**:

- Jika tujuan utama adalah **stabilitas operasional harian** dengan minim false alarm → **Random Forest + class_weight** lebih sesuai (akurasi tinggi, precision baik untuk kelas yang terdeteksi).
- Jika tujuan utama adalah **jangan sampai melewatkan kegagalan tool wear** (misalnya karena biaya kerusakan akibat tool aus sangat mahal) → **Logistic Regression + SMOTE** lebih sesuai untuk kelas spesifik ini, meski perlu menerima lebih banyak false alarm.
- Untuk `Random Failures`, tak satu pun pendekatan klasifikasi berhasil — mengindikasikan kelas ini butuh **pendekatan berbeda** (lihat Rekomendasi).

## Rekomendasi Bisnis & Pengembangan Lanjutan

1. **Gunakan Random Forest + class_weight sebagai model utama sementara** untuk kegagalan yang bisa dideteksi dengan baik (Heat Dissipation, Power, Overstrain), karena keseimbangan akurasi dan precision-nya paling baik untuk operasional harian.
2. **Pertimbangkan model terpisah atau threshold khusus untuk Tool Wear Failure** — mengingat dampak bisnisnya (tool wear tinggi terkait langsung dengan kerusakan mekanis), trade-off menerima Logistic Regression + SMOTE (recall tinggi, precision rendah) untuk kelas ini secara spesifik bisa lebih menguntungkan daripada kehilangan deteksi sepenuhnya.
3. **Alihkan `Random Failures` ke pendekatan anomaly detection**, bukan klasifikasi — karena kegagalannya tampak tidak memiliki pola fitur yang konsisten, model berbasis deviasi dari kondisi normal (mis. Isolation Forest, One-Class SVM) berpotensi lebih efektif.
4. **Lanjutkan hyperparameter tuning** pada Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) untuk mencoba meningkatkan Recall Tool Wear Failure tanpa mengorbankan akurasi keseluruhan secara drastis.
5. **Analisis kesalahan mendalam (error analysis)** pada False Negative Tool Wear Failure dan Random Failures untuk menggali apakah dibutuhkan fitur tambahan (mis. data historis maintenance, usia komponen) di luar sensor real-time yang tersedia saat ini.

## Tools & Library

- **Python 3**
- `pandas`, `numpy` — manipulasi data
- `matplotlib`, `seaborn` — visualisasi (distribusi kelas, korelasi, confusion matrix, feature importance)
- `scikit-learn` — `train_test_split`, `LogisticRegression`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `class_weight`
- `imbalanced-learn` (`imblearn`) — SMOTE


## Struktur Notebook

| Tahap | Isi |
|---|---|
| Kerangka Bisnis | Konteks dan tujuan proyek predictive maintenance |
| EDA | Distribusi target, korelasi fitur sensor, distribusi tipe mesin |
| Data Cleaning | Drop kolom ID, encoding `Type`, cek missing value & duplikasi |
| Feature Engineering | Pembuatan fitur `Power` dan `Temp_Diff` |
| Split & Handling Imbalance | Train-test split berstratifikasi, perhitungan class weight |
| Modeling | 4 kombinasi model: LR/RF × class_weight/SMOTE |
| Feature Importance | Analisis kontribusi fitur dari Random Forest |
| Laporan Analisis | Kesimpulan trade-off model dan rekomendasi bisnis |

## Batasan (Limitations)

- `Random Failures` tidak terdeteksi oleh model apa pun dalam eksperimen ini — kemungkinan besar butuh pendekatan anomaly detection, bukan klasifikasi supervised.
- Precision sangat rendah pada pendekatan SMOTE (khususnya untuk `Tool Wear Failure`) berarti penerapan langsung di lapangan akan menimbulkan banyak false alarm yang perlu dipertimbangkan biayanya terhadap risiko kegagalan yang terlewat.
- Dataset bersifat statis (snapshot), belum mencakup dimensi waktu/urutan kejadian yang bisa memperkaya deteksi pola degradasi mesin secara bertahap (time-series/sequential modeling).
- Belum dilakukan hyperparameter tuning sistematis (mis. GridSearch/RandomizedSearch) — performa model saat ini masih berbasis parameter default/manual.
