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

# Definisikan endpoint untuk prediksi di URL '/predict'
# Hanya menerima request dengan metode POST
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Cek apakah model sudah berhasil dimuat
#     if model_pipeline is None:
#         return jsonify({'error': 'Model tidak tersedia, periksa log server.'}), 500

#     # Ambil data JSON yang dikirim dari client
#     json_data = request.get_json()
#     if not json_data:
#         return jsonify({'error': 'Data tidak ditemukan dalam request'}), 400

#     try:
#         # Konversi data JSON menjadi DataFrame Pandas
#         # Nama kolom di JSON harus sama persis dengan nama kolom di file CSV asli
#         # (sebelum dibersihkan), karena preprocessor akan menanganinya.
#         data_df = pd.DataFrame(json_data, index=[0])

#         # Mengambil probabilitas untuk kelas '1' (Layak Kredit)
#         probabilitas = model_pipeline.predict_proba(data_df)
#         probabilitas_layak = probabilitas[0][1]

#         # Tentukan kelas prediksi berdasarkan ambang batas 0.5
#         hasil_prediksi = 1 if probabilitas_layak >= 0.5 else 0
#         status_teks = 'Layak Kredit' if hasil_prediksi == 1 else 'Tidak Layak Kredit'

#         # Kembalikan hasil dalam format JSON
#         return jsonify({
#             'prediksi': hasil_prediksi,
#             'status_teks': status_teks,
#             'probabilitas_layak': float(probabilitas_layak)
#         })

#     except Exception as e:
#         # Log error yang detail di server untuk debugging internal
#         traceback.print_exc() 
#         # Kirim pesan yang lebih umum ke client
#         return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model tidak tersedia, periksa log server.'}), 500

    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'Data tidak ditemukan dalam request'}), 400

    try:
        data_df = pd.DataFrame(json_data, index=[0])

        # --- BAGIAN YANG DIPERBAIKI ---
        # 1. Gunakan .predict() bukan .predict_proba()
        probabilitas_array = model_pipeline.predict(data_df)
        # 2. Ambil nilai probabilitasnya dari output Keras
        probabilitas_layak = probabilitas_array[0][0]
        # --- AKHIR BAGIAN YANG DIPERBAIKI ---

        hasil_prediksi = 1 if probabilitas_layak >= 0.5 else 0
        status_teks = 'Layak Kredit' if hasil_prediksi == 1 else 'Tidak Layak Kredit'

        return jsonify({
            'prediksi': hasil_prediksi,
            'status_teks': status_teks,
            'probabilitas_layak': float(probabilitas_layak) # Pastikan di-convert ke float
        })

    except Exception as e:
        traceback.print_exc() 
        return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

# Jalankan server Flask jika file ini dieksekusi secara langsung
if __name__ == '__main__':
    # Server akan bisa diakses dari mana saja di jaringan (host='0.0.0.0')
    app.run(host='0.0.0.0', port=5000)