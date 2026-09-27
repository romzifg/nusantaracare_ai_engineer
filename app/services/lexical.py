"""Vektor TF-IDF lokal: tidak membutuhkan model neural atau koneksi internet."""
import math
import re
import unicodedata
from collections import Counter

STOP = set("apa apakah bagaimana berapa saya yang dan atau di ke dari untuk dengan dalam pada itu ini bisa boleh kapan siapa saja harus jika kalau ada tidak sebuah tolong jelaskan mengapa kenapa nya adalah sebagai oleh tersebut agar kepada tentang ingin mau".split())
ALIASES = {
    "wa": "whatsapp", "nyangkut": "gangguan", "lelet": "gangguan",
    "login": "akses akun", "password": "kata sandi", "kredensial": "kata sandi",
    "laptop": "laptop perlengkapan", "surel": "email",
    "minta": "permintaan", "meminta": "permintaan", "mengajukan": "pengajuan",
    "menyetujui": "persetujuan",
    "disetujui": "persetujuan", "setuju": "persetujuan",
    "pulih": "pemulihan", "memulihkan": "pemulihan",
    "lapor": "laporan", "melapor": "laporan", "melaporkan": "laporan",
    "dilaporkan": "laporan", "pelaporan": "laporan",
}


def tokens(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return [term for word in words for term in ALIASES.get(word, word).split()
            if term not in STOP]


def word_count(text):
    """Anggaran chunk dihitung sebagai kata, bukan token model neural."""
    return len(re.findall(r"\S+", text))


class TfidfEncoder:
    def __init__(self, texts, max_features=4096):
        frequency = Counter()
        for text in texts:
            frequency.update(set(tokens(text)))
        # Urutan tetap: proses berbeda menghasilkan posisi dimensi yang sama.
        vocabulary = sorted(sorted(frequency, key=lambda term: (-frequency[term], term))[:max_features])
        if not vocabulary:
            raise ValueError("Dokumen tidak memiliki kosakata untuk pencarian.")
        self.vocabulary = {term: i for i, term in enumerate(vocabulary)}
        self.idf = [math.log((1 + len(texts)) / (1 + frequency[term])) + 1
                    for term in vocabulary]

    def encode(self, text):
        vector = [0.0] * len(self.vocabulary)
        for term, count in Counter(tokens(text)).items():
            position = self.vocabulary.get(term)
            if position is not None:
                vector[position] = (1 + math.log(count)) * self.idf[position]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
