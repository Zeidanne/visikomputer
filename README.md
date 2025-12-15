# Aplikasi Ekstraksi Fitur Tekstur GLCM

Aplikasi web untuk ekstraksi fitur tekstur menggunakan Gray Level Co-occurrence Matrix (GLCM).

## Fitur

-   Upload gambar dalam berbagai format (PNG, JPG, JPEG, BMP, TIFF)
-   Input parameter GLCM yang dapat dikustomisasi:
    -   **Distances**: Jarak piksel (contoh: 1, 2, 3)
    -   **Angles**: Sudut arah dalam derajat (contoh: 0, 45, 90, 135)
    -   **Gray Levels**: Jumlah level abu-abu (8, 16, 32, 64, 128, 256)
-   Ekstraksi 6 fitur tekstur GLCM:
    -   Contrast
    -   Dissimilarity
    -   Homogeneity
    -   Energy
    -   Correlation
    -   ASM (Angular Second Moment)
-   Menampilkan hasil detail per distance dan angle
-   Menampilkan rata-rata setiap fitur

## Instalasi

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:

```bash
python app.py
```

3. Buka browser dan akses:

```
http://localhost:5000
```

## Cara Menggunakan

1. Klik tombol "Pilih Gambar" untuk upload gambar
2. Atur parameter GLCM sesuai kebutuhan:
    - **Distances**: Masukkan jarak piksel (contoh: `1` atau `1,2,3`)
    - **Angles**: Masukkan sudut dalam derajat (contoh: `0,45,90,135`)
    - **Gray Levels**: Pilih jumlah level abu-abu dari dropdown
3. Klik tombol "Ekstraksi Fitur"
4. Hasil akan ditampilkan dalam bentuk tabel yang mencakup:
    - Parameter yang digunakan
    - Rata-rata nilai setiap fitur
    - Detail nilai fitur per distance dan angle

## Teknologi yang Digunakan

-   **Flask**: Web framework
-   **OpenCV**: Image processing
-   **scikit-image**: GLCM feature extraction
-   **NumPy**: Numerical computing

## Tentang GLCM

GLCM (Gray Level Co-occurrence Matrix) adalah metode statistik untuk mengekstraksi fitur tekstur dari gambar dengan menganalisis hubungan spasial antara piksel pada jarak dan sudut tertentu.
