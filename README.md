# Proyek Deteksi Tutup Botol (Ada Mata)

Ini adalah submission saya untuk take-home test Machine Learning Engineer. Proyek ini membangun pipeline ML untuk mendeteksi tutup botol (biru muda, biru tua, lainnya) menggunakan YOLOv8.

## Ringkasan Proyek
* **Model:** YOLOv8n
* **Pipeline:** Dibangun sebagai CLI (alat baris perintah) Python (`bsort`) dengan `typer`.
* **Fitur Utama:** Termasuk skrip `preprocess` otomatis untuk memberi label ulang dataset dari 1 kelas menjadi 3 kelas warna menggunakan OpenCV (analisis HSV).

---

## Hasil & Analisis Eksperimen (V1-V4)
* **Training:** Model dilatih beberapa kali untuk mencoba mendeteksi tutup botol pada gambar tes `...b4_3.jpg`.
* **Link WandB (Publik):** [https://wandb.ai/mewarfarzahrahmawati-universitas-bakrie/ada-mata-bsort](https://wandb.ai/mewarfarzahrahmawati-universitas-bakrie/ada-mata-bsort)
* **Analisis Waktu Inferensi:**
    * Waktu inferensi pada mesin saya (Intel Core i5-8250U CPU) menggunakan **model ONNX** adalah **~84ms - 116ms**.
    * Ekspor TFLite gagal karena *bug* eksternal di *library* `onnx2tf` (error `ai_edge_litert`), sehingga model ONNX yang digunakan untuk *benchmark*.

### Analisis Akar Masalah (Kenapa Deteksi Gagal)
Setelah 4 iterasi eksperimen (V1-V4), model **tetap gagal** mendeteksi tutup botol di gambar tes. Ini **bukan *error* kode**, tetapi **masalah fundamental** pada strategi *preprocessing* (pelabelan otomatis):

Strategi `cv2.mean()` (bahkan dengan *masking* lingkaran di V4) gagal karena:
1.  Tutup botol tes memiliki bagian **[Putih Tengah]** yang dominan.
2.  Saat di-rata-rata dengan **[Cincin Biru Muda]**, hasilnya adalah warna "Putih Kebiruan".
3.  Warna "Putih Kebiruan" ini memiliki nilai *Saturation* yang sangat rendah, sehingga **masih gagal** melewati filter `min_saturation: 30` di `settings.yaml`.
4.  Akibatnya, `preprocess.py` (bahkan V4) **masih salah memberi label** tutup botol ini sebagai `2 (other)`.

Model telah belajar dengan benar untuk mengabaikan objek-objek ini, sesuai dengan data latih yang salah.

### Rekomendasi Perbaikan
Untuk perbaikan, `preprocess.py` harus diubah total dari `cv2.mean()` menjadi strategi yang lebih canggih seperti **K-Means Clustering** (untuk menemukan warna dominan, bukan rata-rata) atau **pelabelan manual**.

---

## Cara Menjalankan

1.  **Setup Environment (Conda):**
    ```bash
    conda create -n adamata python=3.10
    conda activate adamata
    pip install -r requirements.txt
    ```

2.  **Preprocessing Data:**
    *(Pastikan Anda sudah men-download dataset ke `dataset/raw-data/`)*
    ```bash
    python -m bsort.main run-preprocess --config settings.yaml
    ```

3.  **Training Model:**
    *(Pastikan Anda mengisi `wandb_entity` di `settings.yaml`)*
    ```bash
    python -m bsort.main run-train --config settings.yaml
    ```

4.  **Menjalankan Inferensi (Uji Coba):**
    ```bash
    python -m bsort.main run-infer --config settings.yaml --image dataset/raw-data/nama_gambar.jpg
    ```
