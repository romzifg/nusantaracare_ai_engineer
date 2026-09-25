"""Rangkai pencarian dan pemilih bukti menjadi satu agent."""
from app.services.agent import Agent
from app.services.llm import EvidenceSelector
from app.services.retrieval import Retriever


def build_agent(settings):
    return Agent(Retriever(settings), EvidenceSelector(settings), settings)
