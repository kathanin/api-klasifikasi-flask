# Menggunakan image Python versi 3.9 yang ringan
FROM python:3.9-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Instal semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam container
COPY . .

# Perintah untuk menjalankan server Gunicorn saat container dimulai
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]