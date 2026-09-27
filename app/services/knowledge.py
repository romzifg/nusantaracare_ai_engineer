"""Baca dokumen sumber dan tandai bagian aktif serta arsip."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCE_NAME = "nusantaracare_panduan_operasional_internal_v2.md"
SOURCE_SHA256 = "4c8aa6e896425547ec55392a7a8df97142d20e3d93b480aa29494f0e6f514db8"
PIPELINE_VERSION = "nc-ops-v2-title-aware-faq-answer-fastembed-onnx-v1"


@dataclass
class Passage:
    id: str
    text: str
    section: str
    section_id: str
    line_start: int
    line_end: int
    version: str
    active: bool
    authority: str

    def metadata(self, header):
        return {
            "doc_id": header["doc_id"],
            "doc_title": header["doc_title"],
            "doc_version": self.version,
            "effective_date": "2025-01-01" if not self.active else str(header["effective_date"]),
            "effective_until": "2026-06-30" if not self.active else "",
            "is_active": self.active,
            "section_title": self.section,
            "section_id": self.section_id,
            "passage_id": self.id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "authority": self.authority,
        }


def parse_document(path: Path, verify=True):
    raw = path.read_bytes()
    if verify and hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dokumen sumber berubah. Verifikasi dokumen asli, jangan ubah hash sembarang.")
    lines = raw.decode("utf-8-sig").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("Frontmatter wajib tersedia.")
    end = lines.index("---", 1)
    header = yaml.safe_load("\n".join(lines[1:end]))
    if header["doc_id"] != "NC-OPS-001" or str(header["doc_version"]) != "2.0":
        raise ValueError("Identitas sumber tidak sesuai.")
    passages = []
    h2, h3 = "", ""
    start, buffer = None, []

    def flush():
        nonlocal start, buffer
        if buffer:
            text = "\n".join(buffer)
            active = "NONAKTIF" not in h3.upper()
            section = " / ".join(x for x in (h2, h3) if x)
            authority = (
                "archive" if not active else
                "summary" if h2 in {"Lampiran Matriks Keputusan", "Riwayat Perubahan dan Arsip Kebijakan"} else
                "example" if "Contoh" in h3 else
                "faq" if h2 == "FAQ Operasional" else "primary"
            )
            sid = hashlib.sha256(section.encode()).hexdigest()[:12]
            pid = f"NC-{start:04d}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
            passages.append(Passage(pid, text, section, sid, start, start+len(buffer)-1,
                                    "2.0" if active else "1.4", active, authority))
        buffer, start = [], None

    for number, line in enumerate(lines[end+1:], start=end+2):
        if line.startswith("#"):
            flush()
            if line.startswith("## "):
                h2, h3 = line[3:].strip(), ""
            elif line.startswith("### "):
                h3 = line[4:].strip()
        elif not line.strip():
            flush()
        else:
            if start is None:
                start = number
            buffer.append(line)
    flush()
    return header, passages


def make_chunks(passages, header, token_count, max_tokens=112, overlap=18):
    """Potong teks dengan tokenizer FastEmbed; judul ikut dihitung dalam batas token."""
    if max_tokens <= 0 or overlap >= max_tokens or overlap < 0:
        raise ValueError("Ukuran chunk harus positif; overlap harus antara nol dan ukuran chunk.")
    chunks = []
    for p in passages:
        words = list(re.finditer(r"\S+", p.text))
        if not words:
            continue

        start, chunk_number = 0, 0
        while start < len(words):
            low, high, stop = start + 1, len(words), start
            while low <= high:
                middle = (low + high) // 2
                candidate = p.text[words[start].start():words[middle - 1].end()]
                if token_count(f"{p.section}\n{candidate}") <= max_tokens:
                    stop = middle
                    low = middle + 1
                else:
                    high = middle - 1

            # Satu kata sangat panjang tetap dipakai agar proses selalu maju.
            if stop == start:
                single_word = p.text[words[start].start():words[start].end()]
                if token_count(f"{p.section}\n{single_word}") > max_tokens:
                    raise ValueError("Judul atau satu token sumber melebihi batas ukuran chunk.")
                stop = start + 1
            text = p.text[words[start].start():words[stop - 1].end()]
            chunks.append({
                "id": f"{p.id}-c{chunk_number:04d}", "text": text,
                "metadata": p.metadata(header),
            })
            if stop == len(words):
                break
            next_start = stop
            while next_start > start + 1:
                overlap_text = p.text[words[next_start - 1].start():words[stop - 1].end()]
                if token_count(overlap_text) > overlap:
                    break
                next_start -= 1
            start, chunk_number = next_start, chunk_number + 1
    return chunks
