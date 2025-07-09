import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. PEMUATAN DAN PERSIAPAN DATA ---
print("Memuat data mentah...")
df = pd.read_csv('data_nasabah.csv')

# Membuat kolom target biner (0 = Tidak Layak, 1 = Layak)
# Asumsi: kolom target asli bernama 'hasil_klasifikasi' berisi 'Low', 'Medium', 'High'
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

# --- 3. 🧠 MEMBANGUN ARSITEKTUR MODEL NEURAL NETWORK ---
print("Membangun arsitektur model Keras...")
model = tf.keras.Sequential([
    # Input shape akan ditentukan secara otomatis oleh pipeline
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output biner
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. 🤖 MENGGABUNGKAN PREPROCESSOR DAN MODEL ---
# Ini adalah langkah kunci: membuat satu pipeline besar
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])
print("Pipeline final (preprocessor + model) berhasil dibuat.")

# --- 5. MELATIH MODEL ---
# Bagi data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nMemulai training model dengan {len(X_train)} data...")

# Gunakan 'classifier__' untuk memberikan parameter ke bagian model dari pipeline
final_pipeline.fit(X_train, y_train,
                   classifier__epochs=50,
                   classifier__batch_size=32,
                   classifier__verbose=1)
print("Training selesai.")

# --- 6. EVALUASI DAN SIMPAN MODEL FINAL ---
print("\nMengevaluasi model pada data test...")

# 1. Lakukan preprocessing pada data test menggunakan bagian 'preprocessor' dari pipeline
X_test_processed = final_pipeline.named_steps['preprocessor'].transform(X_test)

# 2. Gunakan metode .evaluate() dari model Keras ('classifier') pada data yang sudah diproses
loss, accuracy = final_pipeline.named_steps['classifier'].evaluate(X_test_processed, y_test, verbose=0)

print(f"Akurasi model pada data test: {accuracy * 100:.2f}%")
print(f"Loss model pada data test: {loss:.4f}")

# Simpan keseluruhan pipeline (preprocessor + model) ke satu file
joblib.dump(final_pipeline, 'model_kredit_final.joblib')
print("\n✅ Model final berhasil disimpan sebagai 'model_kredit_final.joblib'")