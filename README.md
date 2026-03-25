# Sistem Deteksi Emosi Teks Indonesia

Aplikasi web untuk klasifikasi emosi pada teks berbahasa Indonesia menggunakan Machine Learning dengan algoritma Naive Bayes MultinomialNB dan Support Vector Machine (SVM).

## Fitur Utama

- Upload data CSV dengan teks Indonesia dan label emosi
- Validasi data minimum 800 baris
- Preprocessing teks bahasa Indonesia (Sastrawi)
- Training dengan 4 skenario berbeda
- Perbandingan 8 model (4 skenario × 2 algoritma)
- Evaluasi otomatis dan pemilihan model terbaik
- Interface prediksi untuk 5 teks sekaligus
- Tampilan hasil yang user-friendly

> **⚠️ Catatan:** Di GitHub Codespaces, ukuran file upload dibatasi **maksimal 10MB** karena keterbatasan proxy platform. Untuk file lebih besar (sampai 100MB), jalankan aplikasi di **local machine** Anda.

## Teknologi yang Digunakan

### Backend
- **Flask** - Web framework
- **scikit-learn** - Machine Learning (MultinomialNB, LinearSVC)
- **pandas** - Data manipulation
- **Sastrawi** - Indonesian text preprocessing
- **joblib** - Model persistence

### Frontend
- **Bootstrap 5** - UI framework
- **Chart.js** - Data visualization
- **Vanilla JavaScript** - Client-side interactivity

## Instalasi

### 1. Clone repository
```bash
cd /workspaces/project-033534355786854
```

### 2. Buat virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000` atau `http://0.0.0.0:5000`

## Cara Penggunaan

### 1. Upload Data
- Siapkan file CSV dengan minimal 800 baris data
- Format CSV harus memiliki kolom untuk **teks** dan **label emosi**
- Contoh label emosi: happy, sadness, anger, fear, love
- Upload file di halaman Upload Data

### 2. Pilih Kolom
- Setelah upload, pilih kolom yang berisi teks
- Pilih kolom yang berisi label emosi
- Klik "Lanjutkan ke Preprocessing"

### 3. Konfigurasi Preprocessing
- Pilih langkah preprocessing yang diinginkan:
  - Lowercase
  - Hapus tanda baca
  - Hapus angka
  - Hapus stopwords (Bahasa Indonesia)
  - Stemming (Sastrawi)
- Preview hasil preprocessing
- Klik "Lanjutkan ke Training"

### 4. Training Model
- Pilih skenario yang ingin dilatih (default semua skenario aktif)
- Setiap skenario melatih 2 model: Naive Bayes dan SVM
- Total 8 model akan dilatih
- Proses training akan memakan waktu beberapa menit
- Klik "Mulai Pelatihan"

### 5. Evaluasi Model
- Lihat tabel perbandingan performa semua model
- Model terbaik ditandai dengan badge hijau "Best"
- Lihat visualisasi grafik perbandingan
- Model terbaik dipilih berdasarkan F1-Score dan Accuracy
- Klik "Lanjut ke Prediksi"

### 6. Prediksi Emosi
- Masukkan hingga 5 teks dalam bahasa Indonesia
- Klik "Prediksi"
- Hasil prediksi ditampilkan dalam tabel dengan badge berwarna per emosi

## Struktur Data CSV

File CSV harus memiliki format berikut:

```csv
text,emotion
"Hari ini aku sangat bahagia sekali!",happy
"Aku merasa sedih dan kecewa",sadness
"Ini membuatku marah",anger
"Aku takut dengan situasi ini",fear
"Aku mencintai keluargaku",love
```

**Persyaratan:**
- Minimal 800 baris data
- Kolom teks berisi kalimat/paragraf dalam bahasa Indonesia
- Kolom emosi berisi label emosi

## Skenario Modeling

### Skenario 1: Konfigurasi Dasar
- Preprocessing: Basic cleaning, lowercase
- Vektorisasi: TF-IDF unigram (3000 features)
- Models: NB (alpha=1.0), SVM (C=1.0)

### Skenario 2: Preprocessing Lanjutan
- Preprocessing: Cleaning, stopwords, stemming
- Vektorisasi: TF-IDF bigram (5000 features)
- Models: NB (alpha=0.5), SVM (C=1.0)

### Skenario 3: Count Vectorizer
- Preprocessing: Cleaning, stopwords, stemming
- Vektorisasi: CountVectorizer bigram (5000 features)
- Models: NB (alpha=1.0), SVM (C=1.0)

### Skenario 4: Konfigurasi Optimal
- Preprocessing: Full pipeline
- Vektorisasi: TF-IDF trigram (7000 features)
- Models: NB (alpha=0.1), SVM (C=10.0)

## Evaluasi Model

Metrik yang digunakan:
- **Accuracy** - Tingkat akurasi keseluruhan
- **Precision** - Rata-rata precision untuk semua kelas
- **Recall** - Rata-rata recall untuk semua kelas
- **F1-Score** - Harmonic mean dari precision dan recall

Model terbaik dipilih berdasarkan **F1-Score** (primary) dan **Accuracy** (secondary).

## Struktur Proyek

```
/workspaces/project-033534355786854/
├── app.py                      # Aplikasi Flask utama
├── config.py                   # Konfigurasi dan skenario
├── requirements.txt            # Dependencies Python
├── .gitignore                  # Git ignore
│
├── models/                     # Model terlatih
│   ├── scenario_*_nb.joblib
│   ├── scenario_*_svm.joblib
│   ├── vectorizer_*.joblib
│   └── best_model_metadata.json
│
├── uploads/                    # File CSV yang diupload
│
├── src/                        # Modul backend
│   ├── data_handler.py
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   ├── model_trainer.py
│   ├── evaluator.py
│   └── predictor.py
│
├── templates/                  # Template HTML
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── preprocessing.html
│   ├── training.html
│   ├── evaluation.html
│   ├── prediction.html
│   └── results.html
│
└── static/                     # File statis
    ├── css/style.css
    └── js/main.js
```

## Troubleshooting

### Error: Sastrawi not found
```bash
pip install Sastrawi
```

### Error: Module not found
Pastikan semua dependencies terinstall:
```bash
pip install -r requirements.txt
```

### Error: File size too large
**Di GitHub Codespaces:** File CSV maksimal 10MB karena keterbatasan proxy nginx platform.

**Untuk file lebih besar (sampai 100MB):** Jalankan aplikasi di local machine Anda:
```bash
git clone <repo-url>
cd project-033534355786854
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Jika perlu mengubah limit di local machine, edit `MAX_CONTENT_LENGTH` di `config.py`.

### Error: Not enough data
Pastikan file CSV memiliki minimal 800 baris data.

## Catatan Penting

- Model akan tersimpan di folder `models/` dan dapat digunakan kembali
- File yang diupload akan tersimpan sementara di folder `uploads/`
- Gunakan tombol "Reset" untuk memulai dari awal dan menghapus session
- Training model membutuhkan waktu beberapa menit tergantung ukuran data

## Author

Sistem Deteksi Emosi Teks Indonesia
© 2026

## License

This project is for educational purposes.
