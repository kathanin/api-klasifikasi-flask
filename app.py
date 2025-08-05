import pandas as pd
import joblib
from flask import Flask, request, jsonify
import traceback
import numpy as np

# --- Blueprint Model (Tetap diperlukan) ---
import tensorflow as tf
from scikeras.wrappers import KerasClassifier

# Fungsi ini HARUS ada di sini agar joblib bisa memuat model Anda.
# Ini adalah "blueprint" arsitektur model Anda.
def create_model(meta):
    n_features_in_ = meta["n_features_in_"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features_in_,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
# --- Akhir Blueprint ---


# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model pipeline final Anda
try:
    model_pipeline = joblib.load("model_kredit_final.joblib")
    print("✅ Model Neural Network berhasil dimuat.")
except Exception as e:
    print(f"🚨 Gagal memuat model: {e}")
    model_pipeline = None

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model tidak tersedia, periksa log server.'}), 500

    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'Data tidak ditemukan dalam request'}), 400

    try:
        data_df = pd.DataFrame(json_data, index=[0])

        # --- BAGIAN YANG DIPERBAIKI SECARA FINAL ---
        
        # 1. Gunakan metode .predict_proba() dari pipeline.
        probabilitas_array = model_pipeline.predict_proba(data_df)
        print(f"DEBUG: Raw output dari Keras model (predict_proba): {probabilitas_array}")
        
        # 2. Ambil probabilitas untuk kelas "1" (Layak). 
        #    Berdasarkan log, outputnya adalah array 1D, jadi kita ambil elemen kedua dengan indeks [1].
        probabilitas_layak = probabilitas_array[1]
            
        # --- AKHIR BAGIAN YANG DIPERBAIKI ---
        
        hasil_prediksi = 1 if probabilitas_layak >= 0.5 else 0
        status_teks = 'Layak Kredit' if hasil_prediksi == 1 else 'Tidak Layak Kredit'

        return jsonify({
            'prediksi': hasil_prediksi,
            'status_teks': status_teks,
            'probabilitas_layak': float(probabilitas_layak)
        })

    except Exception as e:
        traceback.print_exc() 
        return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

# Bagian ini hanya untuk menjalankan server secara lokal
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
