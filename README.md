# ❤️ Sistem Diagnosa Penyakit Jantung dengan Multilayer Perceptron (MLP)
# ❤️ Heart Disease Diagnosis System using Multilayer Perceptron (MLP)

---

## 🇮🇩 Deskripsi Singkat

Aplikasi ini merupakan sistem prediksi risiko penyakit jantung berbasis web yang dibangun menggunakan model *Multilayer Perceptron (MLP)*. Dataset diproses terlebih dahulu dengan pembersihan missing value dan outlier, kemudian dilatih dengan model neural network dan disajikan melalui antarmuka Streamlit.

---

## 🌐 Description (English)

This is a web-based prediction system for heart disease risk built using a trained *Multilayer Perceptron (MLP)*. The data is preprocessed, modeled, and deployed with a clean and interactive UI via Streamlit.

---

## 🧪 Dataset

- 📄 **Raw dataset**: `data/raw_heart_dataset.csv`
- ✅ **Cleaned dataset**: `data/cleaned_heart_dataset.csv` (after preprocessing)

---

## 🔍 Preprocessing (`notebook/preprocessing.ipynb`)

- Analisis kolom, korelasi, dan missing value
- Mengganti nilai 0/NaN dengan modus
- Menangani outlier dengan metode IQR (Capping)
- Menyimpan data bersih ke `cleaned_heart_dataset.csv`

---

## 🧠 Modeling (`notebook/mlp_modeling.ipynb`)

- Fitur numerik → `StandardScaler`
- Fitur kategorikal → `OneHotEncoder`
- Arsitektur MLP: 128-64-32 neuron + Dropout
- Optimizer: AdamW, Loss: BinaryCrossentropy, Metric: AUC
- Callbacks: EarlyStopping, ModelCheckpoint
- Hasil:
  - 📦 Model disimpan → `model/mlp_model.keras`
  - 📦 Preprocessor disimpan → `model/scaler.pkl`

---

## 🖥️ Aplikasi Streamlit (`src/app.py`)

- Input fitur melalui form sidebar
- Otomatis load `mlp_model.keras` dan `scaler.pkl`
- Prediksi risiko penyakit jantung (threshold 0.5)
- Visualisasi hasil & input pengguna

---

## ⚙️ Cara Menjalankan Aplikasi

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi

```bash
streamlit run src/app.py
```
