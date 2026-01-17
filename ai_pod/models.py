"""Shared data models for the ai_pod package."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ArxivCategory(Enum):
    """arXiv category identifiers for CS/ML papers."""

    # Machine Learning & AI
    CS_LG = "cs.LG"      # Machine Learning
    CS_AI = "cs.AI"      # Artificial Intelligence
    CS_CL = "cs.CL"      # Computation and Language (NLP)
    CS_CV = "cs.CV"      # Computer Vision
    CS_NE = "cs.NE"      # Neural and Evolutionary Computing
    CS_IR = "cs.IR"      # Information Retrieval

    # Statistics
    STAT_ML = "stat.ML"  # Machine Learning (Statistics)

    # Other relevant
    CS_RO = "cs.RO"      # Robotics
    CS_HC = "cs.HC"      # Human-Computer Interaction


class OutputMode(Enum):
    """Output mode for paper fetching."""

    TITLE_ONLY = "title_only"
    TITLE_ABSTRACT = "title_abstract"


@dataclass
class Paper:
    """Represents an arXiv paper."""

    arxiv_id: str
    title: str
    abstract: Optional[str]
    authors: list[str]
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str

    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"[{self.arxiv_id}] {self.title}\n  Authors: {authors_str}\n  Published: {self.published.strftime('%Y-%m-%d')}"

