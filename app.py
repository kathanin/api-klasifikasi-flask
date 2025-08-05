import pandas as pd
import joblib
from flask import Flask, request, jsonify
import traceback

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model pipeline final saat aplikasi pertama kali dijalankan
# Ini lebih efisien karena model hanya dimuat sekali.
try:
    model_pipeline = joblib.load("model_kredit_final.joblib")
    print("✅ Model pipeline berhasil dimuat.")
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
        # Konversi data JSON yang diterima menjadi DataFrame
        data_df = pd.DataFrame(json_data, index=[0])

        # --- BAGIAN YANG DIPERBAIKI ---
        # 1. Gunakan .predict_proba() untuk mendapatkan probabilitas
        probabilitas_array = model_pipeline.predict_proba(data_df)
        
        # 2. Ambil probabilitas untuk kelas "1" (Layak), yang berada di indeks ke-1
        probabilitas_layak = probabilitas_array[0][1]
        # --- AKHIR BAGIAN YANG DIPERBAIKI ---

        # Tentukan hasil prediksi kelas berdasarkan ambang batas 0.5
        hasil_prediksi = 1 if probabilitas_layak >= 0.5 else 0
        status_teks = 'Layak Kredit' if hasil_prediksi == 1 else 'Tidak Layak Kredit'

        # Kembalikan hasil dalam format JSON
        return jsonify({
            'prediksi': hasil_prediksi,
            'status_teks': status_teks,
            'probabilitas_layak': float(probabilitas_layak) # Pastikan di-convert ke float
        })

    except Exception as e:
        # Jika terjadi error, cetak traceback untuk debugging dan kembalikan pesan error
        traceback.print_exc() 
        return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

# Bagian ini hanya untuk menjalankan server secara lokal, tidak digunakan oleh Gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
