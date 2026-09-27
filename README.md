# NusantaraCare — Asisten Panduan Layanan Internal

Proyek ini dibuat untuk tugas akhir bootcamp AI Engineer. Tujuannya membantu karyawan menemukan aturan layanan internal tanpa harus membaca seluruh dokumen panduan satu per satu.

Misalnya, karyawan ingin tahu apakah permintaan boleh dikirim lewat WhatsApp. Aplikasi mencari aturan yang berkaitan, memberikan jawaban, lalu menyertakan kutipan agar sumbernya bisa diperiksa.

Aplikasi ini berupa backend: layanan yang menerima pertanyaan dan mengirim jawaban melalui API. Belum ada halaman percakapan seperti ChatGPT, tetapi pengguna bisa mencobanya lewat halaman `/docs`.

## 1. Problem & Success Criteria — Masalah dan Ukuran Keberhasilan

Panduan layanan memuat banyak aturan, termasuk aturan lama yang sudah tidak berlaku. Jika karyawan salah membaca bagian, permintaan bisa dikirim melalui saluran yang keliru atau tidak memenuhi persyaratan.

Asisten ini dibuat untuk mempermudah pencarian informasi dan membantu mengurangi kesalahan tersebut. Pertanyaan yang ditargetkan antara lain:

- “Apakah saya bisa mengirim permintaan melalui WhatsApp?”
- “Apa yang harus dilakukan saat Service Portal tidak bisa diakses?”
- “Siapa yang menyetujui permintaan akses akun?”
- “Berapa hari sebelumnya saya harus meminta perlengkapan?”
- Pertanyaan tentang prioritas gangguan, eskalasi, dan SLA, yaitu target waktu penanganan layanan.

Sistem dianggap memenuhi tujuan jika jawaban sesuai aturan aktif v2.0, memiliki kutipan yang dapat diperiksa, dan tidak memakai arsip v1.4 sebagai aturan saat ini. Bila informasi tidak tersedia, jawaban yang diharapkan adalah **“Tidak ditemukan dalam dokumen.”**

Batas tugasnya adalah menjelaskan panduan. Asisten tidak membuat tiket, mengubah akses akun, atau memeriksa status tiket seseorang secara langsung.

## 2. KB Understanding — Memahami Dokumen Sumber

*Knowledge base* atau KB adalah kumpulan informasi yang menjadi rujukan aplikasi. Dalam proyek ini, sumbernya satu dokumen: [panduan operasional NusantaraCare](data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md).

Isinya mencakup peran petugas, saluran dan jam layanan, prioritas gangguan, akses akun, perlengkapan, kerahasiaan, status tiket, SLA, pertanyaan umum (FAQ), ringkasan aturan, serta arsip kebijakan.

Dokumen memiliki metadata, yaitu keterangan untuk mengenali sumber dan menentukan apakah aturannya masih berlaku.

| Keterangan | Nilai | Artinya |
| --- | --- | --- |
| `doc_id` | `NC-OPS-001` | Identitas dokumen |
| `doc_title` | Panduan Operasional Layanan Internal NusantaraCare | Nama dokumen |
| `doc_version` | `2.0` | Versi kebijakan utama |
| `effective_date` | `2026-07-01` | Mulai berlaku 1 Juli 2026 |
| `last_updated` | `2026-07-15` | Terakhir diperbarui 15 Juli 2026 |
| `is_active` | `true` pada kebijakan utama | Aturan aktif; bagian arsip ditandai nonaktif |

Pembacaan dokumen menghasilkan 102 paragraf, termasuk dua paragraf arsip nonaktif. File sumber dipertahankan apa adanya dan diperiksa menggunakan sidik jari file (SHA-256), sehingga perubahan sumber tidak terlewat begitu saja.

### Perbedaan v1.4 dan v2.0

Arsip v1.4 sudah tidak berlaku sejak 1 Juli 2026. Keberadaannya dalam file yang sama tidak berarti aturannya boleh digunakan kembali.

| Topik | v1.4 — tidak berlaku | v2.0 — aktif |
| --- | --- | --- |
| Saluran permintaan | Email biasa setara dengan portal | Service Portal menjadi saluran utama; email pengganti hanya saat portal tidak tersedia atau tidak dapat diakses |
| Permintaan perlengkapan | Minimal tiga hari kerja sebelumnya | Minimal lima hari kerja sebelumnya |

Dokumen juga memiliki ringkasan yang kurang lengkap dibandingkan prosedur terperinci. Contohnya, matriks akses sementara hanya menyebut persetujuan atasan, sedangkan SOP meminta persetujuan atasan dan pemilik layanan. Karena itu, jawaban mengacu pada SOP atau FAQ aktif, bukan matriks ringkasan maupun arsip.

## 3. RAG Design & Data Preparation — Cara Menyiapkan dan Mencari Informasi

RAG adalah cara menggunakan AI dengan mencari rujukan terlebih dahulu. Dalam proyek ini, AI tidak diminta menjawab hanya berdasarkan pengetahuan bawaannya.

### Menyiapkan potongan dokumen

Dokumen dibaca berdasarkan judul bagian dan paragraf, kemudian paragraf panjang dipecah menjadi potongan kecil (*chunk*). Tujuannya agar pencarian lebih terarah.

- Ukuran maksimal potongan adalah 112 kata, termasuk judul bagiannya. Versi ringan menghitung kata berdasarkan spasi, bukan memakai tokenizer model neural.
- Potongan berurutan dapat mengulang hingga 18 kata (*overlap*) agar konteks di batas potongan tidak langsung hilang.
- Potongan tidak mencampur paragraf atau versi kebijakan yang berbeda. Teks asli tetap disimpan untuk kutipan.
- Setiap potongan memiliki ID dan keterangan sumber, seperti `doc_id`, `doc_title`, `section_title`, `doc_version`, `effective_date`, `is_active`, jenis sumber (`authority`), ID paragraf (`passage_id`), dan nomor baris.

Saat jawaban dikirim, `chunk_id` pada kutipan menunjuk ID paragraf sumber yang digunakan. Ini membantu menelusuri jawaban kembali ke dokumen.

### Menyimpan informasi agar bisa dicari

Teks diubah menjadi vektor TF-IDF: deretan angka yang memberi bobot pada kata. Kata yang khas pada suatu bagian mendapat bobot lebih besar daripada kata yang muncul di banyak bagian. Contohnya, kata “WhatsApp” membantu menemukan aturan saluran layanan.

Versi ini tidak memuat model embedding neural: tidak ada unduhan Hugging Face maupun pemuatan Torch, FastEmbed, atau ONNX Runtime oleh pipeline pencarian. Kosakata dibentuk dari dokumen lokal (maksimal 4096 istilah; dokumen saat ini menghasilkan 1001 dimensi). Alias seperti “wa” → “whatsapp” dan “pulih” → “pemulihan” membantu variasi kata, tetapi ini bukan pemahaman makna seperti model neural.

Vektor disimpan dalam ChromaDB dan dicari berdasarkan cosine similarity. Vector database tetap digunakan, dengan filter versi serta status aktif dokumen. Chroma diberi vektor secara eksplisit dan `embedding_function=None` agar tidak mengunduh model bawaannya. Paket ONNX mungkin masih menjadi dependensi tidak langsung Chroma, tetapi tidak dimuat untuk pencarian ini.

### Mencari sumber yang relevan

Pencarian mengambil hingga delapan kandidat per pencarian (*top-k*), lalu mengurutkannya berdasarkan kemiripan vektor TF-IDF dan kecocokan kata. Hingga enam paragraf teratas digunakan sebagai hasil awal; paragraf prosedur yang berkaitan dapat ditambahkan agar syarat penting tidak terlewat.

Ambang kemiripan awal (*threshold*) adalah `RETRIEVAL_MIN_SCORE=0.12`. Hasil di bawah nilai tersebut tidak digunakan. Skala TF-IDF berbeda dari model lama, sehingga `MIN_SIMILARITY=0.35` dari versi ONNX tidak dipakai lagi. Angka ini bukan persentase kebenaran dan masih perlu evaluasi pada pertanyaan baru.

Pencarian dibatasi ke `doc_version=2.0`, `is_active=true`, dan sumber SOP/FAQ. Bagian arsip, contoh, dan matriks ringkasan tidak dijadikan dasar jawaban.

### Menentukan jawaban

Instruksi kepada model adalah menggunakan konteks dokumen aktif saja, mengutamakan SOP, dan tidak menjawab jika buktinya tidak cukup. Model memilih ID paragraf yang mendukung jawaban, bukan bebas menulis fakta baru. Backend memeriksa pilihan tersebut sebelum menyusun jawaban dari teks sumber.

Jika pertanyaan cocok langsung dengan FAQ, sistem dapat menggunakan jawaban FAQ tanpa memanggil model bahasa. Model bahasa (*LLM*) untuk pertanyaan lain bisa menggunakan Ollama atau layanan hosted seperti OpenRouter.

Catatan privasi: embedding tetap lokal, tetapi jika memakai hosted, pertanyaan dan potongan dokumen yang ditemukan akan dikirim ke penyedia model untuk pemilihan bukti. Gunakan Ollama di lingkungan sendiri jika proses ini juga harus tetap internal.

## 4. Arsitektur — Alur Kerja dan Susunan Folder

Ada dua alur utama:

```text
Saat aplikasi mulai:
Dokumen → dibaca dan dipotong → dibuat vektor TF-IDF → disimpan di ChromaDB

Saat pengguna bertanya:
Pertanyaan → pemeriksaan input → pencarian aturan aktif
          → pemilihan dan pemeriksaan bukti → jawaban beserta kutipan
```

Istilah *agent* pada proyek ini berarti pengatur langkah pencarian dan pemeriksaan tersebut. Agent tidak diberi kebebasan menjalankan perintah atau mengubah sistem lain.

| Lokasi | Kegunaan |
| --- | --- |
| `app/main.py` | Pintu masuk API: menerima pertanyaan, memeriksa token akses, dan mengatur respons/error |
| `app/config.py` | Membaca pengaturan, seperti model, alamat layanan, dan batas permintaan |
| `app/schemas.py` | Menentukan bentuk data pertanyaan dan jawaban |
| `app/services/knowledge.py` | Membaca dokumen, menandai arsip, dan membuat potongan teks |
| `app/services/retrieval.py` | Menyimpan vektor dan mencari sumber yang relevan di ChromaDB |
| `app/services/lexical.py` | Mengubah teks menjadi vektor TF-IDF lokal dan menyamakan beberapa variasi kata |
| `app/services/runtime_stats.py` | Mencatat waktu dan RAM proses pada tahap startup, tanpa mencatat pertanyaan atau secret |
| `app/services/llm.py` | Berkomunikasi dengan Ollama atau layanan hosted |
| `app/services/agent.py` | Mengatur pemilihan sumber, penolakan, dan penyusunan jawaban |
| `app/services/rag.py` | Menghubungkan komponen pencarian dan model saat aplikasi dimulai |
| `data/raw_docs/` | Menyimpan dokumen asli sebagai rujukan |
| `reports/` | Menyimpan hasil evaluasi, bukan sumber jawaban pengguna |
| `docs/` | Menyimpan penjelasan tambahan dan laporan pengujian |
| `tests/` | Menyimpan kode pengujian otomatis |
| `.cache/` | Menyimpan indeks pencarian; cache model dari versi lama tidak dipakai dan tidak perlu diunggah |

## 5. Kontrak API — Cara Mengirim Pertanyaan dan Membaca Jawaban

Kontrak API adalah kesepakatan bentuk data yang dikirim dan diterima aplikasi.

| Alamat | Fungsi |
| --- | --- |
| `POST /ask` | Mengirim pertanyaan |
| `GET /health` | Memeriksa apakah proses API berjalan |
| `GET /ready` | Memeriksa kesiapan indeks pencarian; belum memeriksa koneksi model bahasa |
| `GET /docs` | Membuka dokumentasi interaktif untuk mencoba API |

Contoh isi permintaan ke `/ask`:

```json
{
  "question": "Apakah saya bisa mengirim permintaan melalui WhatsApp?"
}
```

`question` harus berupa teks sepanjang 3–1500 karakter. Jangan menambahkan kolom lain dalam permintaan.

| Kolom jawaban | Cara membacanya |
| --- | --- |
| `answer` | Jawaban atau keterangan mengapa pertanyaan tidak dapat dijawab |
| `confidence_label` | Label `high`, `medium`, atau `low` berdasarkan aturan sederhana, bukan persentase kepastian |
| `reason_code` | Alasan hasil, misalnya `answered` atau `no_relevant_context` |
| `citations` | Kutipan sumber: ID, judul, versi, tanggal berlaku, status aktif, bagian, nomor baris, dan teks kutipan |
| `route` | Jalur penanganan pertanyaan yang dipilih sistem |
| `mode` | Cara pemrosesan yang digunakan, misalnya `ollama` atau `hosted` |
| `trace` | Catatan singkat tahapan pemrosesan untuk membantu pemeriksaan |

Jika `API_ACCESS_TOKEN` diisi, permintaan `/ask` harus menyertakan header `X-API-Key` dengan nilai token tersebut. Token ini melindungi API sendiri; berbeda dari `LLM_API_KEY` yang dipakai untuk menghubungi penyedia model.

HTTP `200` berarti permintaan selesai diproses, tetapi jawabannya bisa berupa penolakan. Periksa juga `reason_code`. Kode lain yang perlu dikenali: `401` token tidak cocok, `413` data terlalu besar, `422` format tidak valid, `429` layanan sedang penuh, `502` keluaran model tidak valid, `503` indeks/model belum tersedia, dan `500` kesalahan internal.

## 6. Cara Menjalankan Lokal

Langkah berikut menggunakan PowerShell dari folder `final_project`. Gunakan Python 3.11 sesuai `pyproject.toml`; verifikasi versi ringan dilakukan dengan Python 3.11.15.

### Siapkan lingkungan Python

Jalankan saat pertama kali menyiapkan proyek. Jika `.venv` sudah tersedia, gunakan lingkungan tersebut.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Buat `.env` dari contoh jika belum ada. Perintah ini tidak menimpa pengaturan yang sudah disimpan:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

### Pilih layanan model

Pilih salah satu pengaturan berikut di `.env`; pengaturan pencarian lainnya bisa mengikuti `.env.example`.

**Pilihan A: Ollama di komputer sendiri.** Pasang dan jalankan Ollama, lalu unduh model dengan `ollama pull llama3`. Pilihan ini tidak membutuhkan API key penyedia model, tetapi membutuhkan memori komputer untuk menjalankan model.

```dotenv
APP_ENV=local
LLM_PROVIDER=ollama
LLM_MODEL=llama3:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

**Pilihan B: layanan hosted melalui OpenRouter.** Isi API key dan nama model yang dapat diakses oleh akun Anda. `openrouter/free` adalah contoh pada konfigurasi proyek; ketersediaan model dan kuota mengikuti penyedia layanan.

```dotenv
APP_ENV=local
LLM_PROVIDER=hosted
LLM_MODEL=openrouter/free
HOSTED_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=isi_api_key_anda
```

Jangan membagikan isi `.env` atau memasukkannya ke repository publik.

### Jalankan dan coba API

```powershell
.\.venv\Scripts\Activate.ps1
fastapi dev
```

Jika `fastapi dev` mengarah ke venv aplikasi lain, gunakan executable proyek secara langsung:

```powershell
.\.venv\Scripts\fastapi.exe dev
```

Perintah `fastapi dev` membaca lokasi aplikasi dari `pyproject.toml` dan memuat ulang server saat kode berubah. Pada pemakaian pertama, aplikasi membuat indeks TF-IDF tanpa unduhan model. `MAX_CONCURRENT=1` membatasi satu pertanyaan aktif per proses agar beban lebih terkendali.

Buka [dokumentasi API lokal](http://127.0.0.1:8000/docs), pilih `POST /ask`, klik **Try it out**, isi pertanyaan, lalu klik **Execute**. Jika token akses diaktifkan, isi token lewat tombol **Authorize** terlebih dahulu.

Untuk menghentikan server, tekan `Ctrl+C`. Untuk keluar dari lingkungan Python, jalankan `deactivate`.

## 7. Deployment — Menjalankan Aplikasi di Cloud

Percobaan FastAPI Cloud dengan model embedding neural, termasuk ONNX, gagal karena kehabisan memori (*out of memory* atau OOM). Versi sekarang memakai TF-IDF lokal + ChromaDB untuk menghilangkan kebutuhan memuat model tersebut. Uji lokal sekitar 109 MiB RAM, tetapi keberhasilan deploy versi TF-IDF **belum diverifikasi**; penggunaan RAM Linux/cloud bisa berbeda dari Windows.

Hal yang perlu disiapkan sebelum mencoba deploy kembali:

1. Pastikan folder aplikasi yang dipilih memuat `pyproject.toml`, dependensi, folder `app`, dan dokumen pada `data/raw_docs`. Jangan ikut mengunggah `.env`, `.venv`, atau cache lokal.
2. Masukkan pengaturan melalui environment/secrets cloud. Gunakan `APP_ENV=production` dan `API_ACCESS_TOKEN` acak minimal 32 karakter.
3. Pilih layanan model. Untuk hosted, isi `LLM_PROVIDER=hosted`, `LLM_MODEL`, `HOSTED_BASE_URL` HTTPS, dan `LLM_API_KEY`. Untuk Ollama, sediakan server Ollama yang dapat dijangkau dengan aman dari cloud. `localhost` di cloud bukan komputer pribadi Anda.
4. Gunakan `RETRIEVAL_MIN_SCORE=0.12` dan `MAX_CONCURRENT=1`. Hapus pengaturan lama `EMBEDDING_MODEL`, `EMBEDDING_CACHE`, dan `MIN_SIMILARITY` dari konfigurasi cloud agar tidak membingungkan; kode baru mengabaikannya. Tidak diperlukan `HF_TOKEN` untuk pencarian TF-IDF. Jalankan satu worker server, tanpa `--reload` di cloud; batas concurrency berlaku per proses.
5. Setelah aplikasi aktif, periksa `/health`, `/ready`, dan `/ask`. Uji jawaban yang ada di dokumen, pertanyaan yang tidak ada, penolakan token salah, serta aturan aktif v2.0.

Pantau penggunaan memori saat aplikasi mulai dan ketika menerima pertanyaan. Status `/ready` saja belum membuktikan bahwa koneksi ke penyedia model sudah berhasil.

Log baru menampilkan `startup stage=begin`, `retrieval_imported`, `tfidf_ready`, `chroma_opened`, `index_ready`, dan `ready`, beserta RAM proses. Pastikan build memakai commit terbaru dan log mencapai `Application startup complete`. TF-IDF menggunakan koleksi Chroma baru; indeks ONNX lama tidak dihapus atau dicampur. Indeks yang terputus di tengah pembuatan dapat dilanjutkan.

Uji lokal tanpa memanggil layanan LLM:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.verify_lightweight --output reports/hasil-tfidf-baru.json
```

## 8. Keterbatasan

- TF-IDF bergantung pada kecocokan kata. Pertanyaan dengan sinonim yang belum dikenali dapat terlewat. Vector database tetap digunakan, tetapi ini bukan neural semantic embedding; jika rubrik mewajibkan model embedding neural tertentu, desain ini perlu dikonfirmasi kepada pengajar.
- Jawaban terbatas pada isi dokumen. Asisten tidak mengetahui nomor, status, atau informasi lain yang tidak dicantumkan dalam panduan.
- Kutipan diperiksa terhadap sumber, tetapi model masih bisa memilih paragraf yang kurang tepat. Jawaban juga dapat terasa panjang karena mempertahankan teks sumber.
- Pemeriksaan instruksi berbahaya masih berbasis pola, sehingga belum menjamin semua upaya manipulasi pertanyaan dapat dikenali.
- Pengujian memakai pertanyaan dalam jumlah terbatas. Hasil baik pada kumpulan tersebut belum membuktikan semua pertanyaan baru akan dijawab dengan benar.
- Belum ada integrasi tiket, login akun perusahaan, maupun pemantauan operasional yang lengkap. Kapasitas banyak pengguna dan batas memori cloud masih perlu diuji.
- Saat memakai hosted, pertanyaan dan konteks dokumen dikirim ke penyedia model. Penggunaannya perlu mengikuti aturan kerahasiaan organisasi.

## 9. Kesimpulan & Rekomendasi

Proyek ini sudah menyediakan alur pencarian panduan, penyaringan kebijakan aktif, dan jawaban dengan kutipan sumber. Manfaat utamanya adalah membantu pengguna menemukan aturan sekaligus mengetahui asal jawabannya. Sistem tetap perlu diuji lebih jauh sebelum diandalkan untuk layanan sehari-hari.

### Ringkasan performa yang sudah diperiksa

Hasil versi lama dan versi sekarang dipisahkan karena mesin embedding telah berubah.

| Pengujian | Hasil | Maknanya |
| --- | --- | --- |
| Versi lama, PyTorch + Ollama, 24 September 2026 | 24 dari 24 kasus evaluasi lolos | Hasil historis pada kumpulan pertanyaan yang sudah dikenal, bukan jaminan untuk versi ONNX |
| Versi ONNX, 27 September 2026 | 149 potongan dokumen berhasil diindeks | Dokumen berhasil disiapkan untuk pencarian lokal |
| Pencarian versi ONNX | 23 dari 23 pertanyaan menemukan bagian yang diharapkan dalam enam hasil teratas | Menguji pencarian sumber, bukan ketepatan jawaban akhir AI |
| API lokal versi ONNX | `/health`, `/ready`, dan dua pertanyaan `/ask` mendapat HTTP 200 | Diuji memakai mode `extractive`, yaitu pengambilan kutipan tanpa model bahasa |
| Deployment cloud ONNX | Verification Failed (OOM) | ONNX saja belum cukup mengatasi batas memori instance |
| TF-IDF lokal + ChromaDB | 108 potongan, 1001 dimensi; 23/23 kasus pencarian lama lolos setelah normalisasi kata | Pencarian sumber saja, bukan akurasi jawaban LLM atau pengujian independen |
| RAM dan startup TF-IDF lokal | Sekitar 109 MiB RAM proses; sekitar 5 detik termasuk import pada uji Windows | Bukan pengukuran di Linux/cloud, bukan uji beban banyak pengguna |
| Unit/API tests TF-IDF | 17/17 lolos | Termasuk filter sumber aktif, autentikasi dan pemulihan indeks parsial; pemilihan LLM sebagian memakai tiruan |
| Deployment cloud TF-IDF | Belum dijalankan | Perlu commit, push, deploy ulang, dan uji endpoint publik |

Ada catatan penting: dari 13 pertanyaan yang seharusnya ditolak, tujuh masih mendapatkan kandidat hasil pencarian. Ini belum berarti sistem memberikan jawaban salah, tetapi keputusan menolak harus diperiksa lagi bersama model bahasa.

Jadi, angka 23/23 tidak berarti akurasi jawaban 100%. Uji pertama TF-IDF mendapat 22/23; perbaikan dilakukan pada kumpulan yang sama. Hasil percobaan tetap disimpan dan hasil kode akhir ada di [laporan verifikasi TF-IDF](reports/lightweight-tfidf-release.json). Evaluasi LLM nyata belum diulang untuk TF-IDF. Rincian tersedia dalam [laporan pengujian lokal](docs/LAPORAN_PENGUJIAN.md).

### Perbaikan berikutnya

1. **Uji alur lengkap versi TF-IDF bersama model bahasa.** Periksa jawaban, kutipan, dan penolakan, terutama untuk informasi yang tidak ada di dokumen.
2. **Tambahkan pertanyaan baru.** Sertakan bahasa sehari-hari, salah ketik, pertanyaan gabungan, dan pertanyaan yang mirip tetapi membutuhkan aturan berbeda. Minta pemilik SOP menilai ketepatan jawaban.
3. **Ukur memori dan waktu respons di cloud.** Periksa saat pemuatan awal maupun saat melayani pertanyaan sebelum menentukan kapasitas server.
4. **Perbaiki pencarian dan keterbacaan jawaban berdasarkan hasil uji.** Sesuaikan ambang pencarian bila diperlukan, kurangi pengulangan, dan tetap sertakan bukti yang mendukung jawaban.
5. **Lengkapi pengamanan sebelum dipakai lebih luas.** Tambahkan pembatasan penggunaan, pemantauan error tanpa menyimpan rahasia, serta pengaturan akses sesuai kebutuhan organisasi.
