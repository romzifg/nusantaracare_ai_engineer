"""Pemeriksaan pola berisiko; tidak semua serangan bisa dikenali."""
import re
import unicodedata


def normalize(text):
    return re.sub(r"\s+", " ", "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(c) != "Cf"
    )).strip()


def refusal_reason(question):
    q = normalize(question)
    patterns = [
        r"(abaikan|ignore|lupakan|disregard|bypass).{0,60}(instruksi|aturan|previous|prompt|policy|safety)",
        r"(tampilkan|bocorkan|ungkap|cetak|show|reveal|print).{0,45}(system prompt|prompt sistem|api.key|kunci api|rahasia|secret)",
        r"(system|developer)\s*:",
        r"(jailbreak|developer mode|dan mode)",
        r"(jalankan|execute|run).{0,30}(shell|powershell|curl|command|perintah)",
        r"(pakai|gunakan|ikuti).{0,40}(v1\.4|versi 1\.4).{0,40}(aktif|berlaku|sekarang)",
    ]
    if any(re.search(p, q) for p in patterns):
        return "unsafe_request"
    if re.search(r"(password|kata sandi|api[_ -]?key|token|kode mfa)\s*[:=]\s*\S+", q):
        return "sensitive_input"
    if re.search(r"\b(sk-[a-zA-Z0-9_-]{12,}|gsk_[a-zA-Z0-9]{12,})\b", question):
        return "sensitive_input"
    if re.search(r"(lihat|tampilkan|buka|berikan).{0,40}(isi|status|data).{0,25}tiket.{0,40}(budi|siti|rekan|orang lain|karyawan lain)", q):
        return "private_data_request"
    if re.search(r"(hitung|berapa|jumlah).{0,25}(gaji|tunjangan|kompensasi)|diagnos|obat untuk|resep obat|nasihat hukum|harga saham|ramalan|cuaca|resep nasi", q):
        return "out_of_scope"
    # API hanya mencari informasi; ia tidak melakukan tindakan pada tiket.
    if re.search(r"^(tolong )?(buatkan|buat|tutup|hapus|eskalasikan|kirim) (tiket|email|laporan)", q):
        return "action_not_supported"
    return None
