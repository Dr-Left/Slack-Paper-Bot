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
class Author:
    """Represents a paper author with optional affiliation."""

    name: str
    affiliation: Optional[str] = None

    def __str__(self) -> str:
        if self.affiliation:
            return f"{self.name} ({self.affiliation})"
        return self.name


@dataclass
class Paper:
    """Represents an arXiv paper."""

    arxiv_id: str
    title: str
    abstract: Optional[str]
    authors: list[Author]
    published: datetime
    updated: datetime
    categories: list[str]
    pdf_url: str

    def __str__(self) -> str:
        author_names = [a.name for a in self.authors[:3]]
        authors_str = ", ".join(author_names)
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"[{self.arxiv_id}] {self.title}\n  Authors: {authors_str}\n  Published: {self.published.strftime('%Y-%m-%d')}"


@dataclass
class PastPaper:
    """A paper from user's reading history."""

    title: str
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None


@dataclass
class UserProfile:
    """Researcher's interest profile for paper filtering."""

    topics: list[str]  # Natural language topic descriptions
    keywords: list[str]  # Specific keywords to boost
    past_papers: list[PastPaper]  # Papers user found interesting
    preferred_authors: Optional[list[str]] = None  # Optional author preferences
    name: str = "default"  # Profile identifier


@dataclass
class FilteredPaper:
    """Paper with its similarity score."""

    paper: Paper
    similarity_score: float

    def __str__(self) -> str:
        return f"[{self.similarity_score:.3f}] {self.paper}"


