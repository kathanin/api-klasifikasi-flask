import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- 1. PEMUATAN DAN PERSIAPAN DATA ---
print("Memuat data mentah...")
df = pd.read_csv('data_nasabah.csv')

# Membuat kolom target biner
target_map = {'Low': 1, 'Medium': 1, 'High': 0}
df['target'] = df['hasil_klasifikasi'].map(target_map)
df = df.drop(columns=['hasil_klasifikasi'])

# Pisahkan Fitur (X) dan Target (y)
X = df.drop('target', axis=1)
y = df['target']
print("Persiapan data selesai.")

# --- 2. MEMBUAT PIPELINE PREPROCESSING ---
kolom_numerik = [
    'umur', 'jumlah_tanggungan', 'jumlah_penghasilan',
    'jumlah_tabungan', 'jumlah_pengajuan', 'tenor'
]
kolom_kategorikal = [
    'status_pernikahan', 'pekerjaan', 
    'riwayat_kredit', 'tujuan_kredit'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), kolom_numerik),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), kolom_kategorikal)
    ],
    remainder='passthrough'
)

# --- 3. 🧠 MEMBANGUN ARSITEKTUR MODEL (DI DALAM FUNGSI) ---
# DIUBAH: Arsitektur model kini didefinisikan di dalam sebuah fungsi
def create_model(meta):
    # Dapatkan jumlah fitur input dari metadata scikeras
    n_features_in_ = meta["n_features_in_"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features_in_,)), # Definisikan input shape dengan benar
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # Output biner dengan probabilitas
    ])
    return model

# --- 4. 🤖 MENGGABUNGKAN PREPROCESSOR DAN MODEL ---
print("Membangun pipeline final...")

# DIUBAH: Bungkus fungsi model menggunakan KerasClassifier
# Parameter training (epochs, batch_size) didefinisikan di sini
keras_model = KerasClassifier(
    model=create_model,
    optimizer="adam",
    loss="binary_crossentropy",
    epochs=50,
    batch_size=32,
    verbose=1,
    metrics=['accuracy']
)

# Buat pipeline besar dengan model yang sudah dibungkus
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', keras_model) # Gunakan model yang sudah dibungkus
])
print("Pipeline final (preprocessor + model) berhasil dibuat.")

# --- 5. MELATIH MODEL ---
# Bagi data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nMemulai training model dengan {len(X_train)} data...")

# Hitung bobot kelas secara otomatis berdasarkan ketidakseimbangan data training.
# Ini akan memberikan bobot lebih tinggi pada kelas minoritas ('High').
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print(f"INFO: Menggunakan Class Weights untuk menyeimbangkan data: {class_weight}")

# DIUBAH: Tambahkan parameter classifier__class_weight ke dalam .fit()
# Ini memberitahu Keras untuk lebih memperhatikan kelas 'High' (kelas 0).
final_pipeline.fit(X_train, y_train, classifier__class_weight=class_weight)

# # Cukup jalankan .fit() pada pipeline. Parameter training sudah didefinisikan di atas.
# final_pipeline.fit(X_train, y_train)
# print("Training selesai.")

# --- 6. EVALUASI DAN SIMPAN MODEL FINAL ---
print("\nMengevaluasi model pada data test...")
accuracy = final_pipeline.score(X_test, y_test)
print(f"Akurasi model pada data test: {accuracy * 100:.2f}%")

# Simpan keseluruhan pipeline ke satu file
joblib.dump(final_pipeline, 'model_kredit_final.joblib')
print("\n✅ Model final berhasil disimpan sebagai 'model_kredit_final.joblib'")
