import hashlib
from app.config import ROOT
from app.services.knowledge import SOURCE_NAME, SOURCE_SHA256, parse_document

path = ROOT / "data/raw_docs" / SOURCE_NAME
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == SOURCE_SHA256, "Hash dokumen sumber tidak cocok."
header, passages = parse_document(path)
print("SHA256:", digest)
print("Paragraf:", len(passages), "Arsip nonaktif:", sum(not p.active for p in passages))
