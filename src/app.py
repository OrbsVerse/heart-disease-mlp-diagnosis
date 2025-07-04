import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# --- 1. KONFIGURASI APLIKASI ---
# Path ke file model dan preprocessor
PREPROCESSOR_PATH = 'scaler.pkl'
MODEL_PATH = 'mlp_model.keras'

# Definisi kolom fitur sesuai urutan saat training
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
ALL_FEATURE_COLS_ORDER = NUMERICAL_COLS + CATEGORICAL_COLS

# Opsi dan mapping untuk input kategorikal (memisahkan data dari UI)
SEX_MAP = {"Wanita": 0, "Pria": 1}
CP_MAP = {
    "Typical Angina (Tipe 1)": 1,
    "Atypical Angina (Tipe 2)": 2,
    "Non-anginal Pain (Tipe 3)": 3,
    "Asymptomatic (Tipe 4)": 4
}
FBS_MAP = {"< 120 mg/dl (Salah)": 0, "> 120 mg/dl (Benar)": 1}
RESTECG_MAP = {
    "Normal (0)": 0,
    "Kelainan Gelombang ST-T (1)": 1,
    "Hipertrofi Ventrikel Kiri (2)": 2
}
EXANG_MAP = {"Tidak": 0, "Ya": 1}
SLOPE_MAP = {"Upsloping (Naik)": 1, "Flat (Datar)": 2}

# --- 2. FUNGSI-FUNGSI PEMBANTU ---

@st.cache_resource # Cache resource agar tidak perlu load ulang pada setiap interaksi
def load_artifacts(preprocessor_path, model_path):
    """Memuat preprocessor dan model yang telah dilatih."""
    try:
        preprocessor = joblib.load(preprocessor_path)
        model = tf.keras.models.load_model(model_path)
        return preprocessor, model
    except FileNotFoundError:
        st.error(f"Error: File tidak ditemukan. Pastikan '{preprocessor_path}' dan '{model_path}' ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat artefak: {e}")
        st.stop()

def get_user_inputs():
    """Membuat dan mengelola semua widget input pengguna di sidebar dalam sebuah form."""
    st.sidebar.header("Input Data Pasien")
    
    with st.sidebar.form(key='prediction_form'):
        st.subheader("Data Umum & Klinis")
        age = st.number_input("Usia (tahun)", 1, 120, 50)
        trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", 50, 250, 120)
        chol = st.number_input("Kolesterol Serum (mg/dl)", 100, 600, 200)
        thalach = st.number_input("Detak Jantung Maksimum (denyut/menit)", 50, 250, 150)
        oldpeak = st.number_input("ST Depression akibat Latihan", 0.0, 10.0, 1.0, 0.1)

        st.subheader("Data Kategorikal")
        sex_label = st.selectbox("Jenis Kelamin", options=list(SEX_MAP.keys()), index=1)
        cp_label = st.selectbox("Tipe Nyeri Dada", options=list(CP_MAP.keys()), index=3)
        fbs_label = st.selectbox("Gula Darah Puasa > 120 mg/dl?", options=list(FBS_MAP.keys()))
        restecg_label = st.selectbox("Hasil EKG Istirahat", options=list(RESTECG_MAP.keys()))
        exang_label = st.selectbox("Angina Akibat Latihan?", options=list(EXANG_MAP.keys()))
        slope_label = st.selectbox("Slope Puncak Latihan ST", options=list(SLOPE_MAP.keys()), index=1)
        
        submit_button = st.form_submit_button(label='Prediksi Risiko')

    input_data = {
        'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
        'sex': SEX_MAP[sex_label], 'cp': CP_MAP[cp_label], 'fbs': FBS_MAP[fbs_label],
        'restecg': RESTECG_MAP[restecg_label], 'exang': EXANG_MAP[exang_label], 'slope': SLOPE_MAP[slope_label]
    }
    
    return input_data, submit_button

def make_prediction(preprocessor, model, input_data, feature_order):
    """Melakukan preprocessing dan prediksi berdasarkan input pengguna."""
    try:
        input_df = pd.DataFrame([input_data])[feature_order]
        processed_input = preprocessor.transform(input_df)
        prediction_proba = model.predict(processed_input)
        return prediction_proba[0][0], input_df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        return None, None

def display_results(probability, raw_input_df):
    """Menampilkan hasil prediksi kepada pengguna."""
    st.subheader("Hasil Prediksi")
    
    col1, col2 = st.columns(2)
    col1.metric("Probabilitas Penyakit Jantung", f"{probability:.2%}")
    
    threshold = 0.5
    if probability >= threshold:
        col2.error("Risiko Tinggi")
        st.error("Berdasarkan data yang dimasukkan, pasien diprediksi **memiliki** risiko tinggi terkena penyakit jantung. Disarankan untuk konsultasi lebih lanjut dengan profesional medis.")
    else:
        col2.success("Risiko Rendah")
        st.success("Berdasarkan data yang dimasukkan, pasien diprediksi **memiliki** risiko rendah terkena penyakit jantung. Tetap jaga pola hidup sehat.")
        
    with st.expander("Lihat Data Input yang Digunakan untuk Prediksi"):
        st.dataframe(raw_input_df)

# --- 3. ALUR UTAMA APLIKASI ---
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide", initial_sidebar_state="expanded")
    
    preprocessor, model = load_artifacts(PREPROCESSOR_PATH, MODEL_PATH)

    st.title("🩺 Prediksi Risiko Penyakit Jantung")
    st.markdown("Aplikasi ini menggunakan model *Neural Network* untuk memprediksi kemungkinan adanya penyakit jantung berdasarkan data klinis pasien. Silakan masukkan data di sidebar kiri.")
    st.info("**Disclaimer:** Hasil prediksi ini bukan diagnosis medis. Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk evaluasi kesehatan yang akurat.", icon="⚠️")

    user_input, submitted = get_user_inputs()
    
    if submitted:
        prediction_probability, raw_df = make_prediction(preprocessor, model, user_input, ALL_FEATURE_COLS_ORDER)
        if prediction_probability is not None:
            display_results(prediction_probability, raw_df)

if __name__ == '__main__':
    main()
