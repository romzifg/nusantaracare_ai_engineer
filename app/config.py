"""Pengaturan aplikasi dari environment atau file .env."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    provider: str = "hosted"
    model: str = "openrouter/free"
    ollama_url: str = "http://127.0.0.1:11434"
    hosted_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    api_token: str = ""
    environment: str = "local"
    db_path: str = str(ROOT / ".cache" / "chroma")
    min_similarity: float = 0.12
    top_k: int = 8
    timeout: float = 90.0
    max_concurrent: int = 1

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        value = cls(
            provider=os.getenv("LLM_PROVIDER", "hosted"),
            model=os.getenv("LLM_MODEL", "openrouter/free"),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            hosted_url=os.getenv("HOSTED_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            api_token=os.getenv("API_ACCESS_TOKEN", ""),
            environment=os.getenv("APP_ENV", "local"),
            db_path=os.getenv("CHROMA_PATH", str(ROOT / ".cache" / "chroma")),
            # Skala TF-IDF berbeda dari ONNX. Jangan pakai threshold lama dari .env.
            min_similarity=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.12")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT", "1")),
        )
        value.validate()
        return value

    def validate(self):
        if self.environment not in {"local", "production"}:
            raise ValueError("APP_ENV harus local atau production; salah ketik tidak boleh melewati proteksi production.")
        if self.provider not in {"ollama", "hosted", "extractive"}:
            raise ValueError("LLM_PROVIDER harus ollama, hosted, atau extractive.")
        if not 0 <= self.min_similarity <= 1:
            raise ValueError("RETRIEVAL_MIN_SCORE harus antara 0 dan 1.")
        if self.max_concurrent < 1:
            raise ValueError("MAX_CONCURRENT minimal 1.")
        if self.environment == "production":
            if len(self.api_token) < 32:
                raise ValueError("Production memerlukan API_ACCESS_TOKEN minimal 32 karakter.")
            if self.provider == "extractive":
                raise ValueError("Mode extractive hanya baseline lokal, bukan submission GenAI.")
        if self.provider == "hosted":
            if not self.llm_api_key or not self.hosted_url.startswith("https://"):
                raise ValueError("Hosted memerlukan LLM_API_KEY dan URL HTTPS.")
