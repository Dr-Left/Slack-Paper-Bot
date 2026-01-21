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
    CS_DC = "cs.DC"      # Distributed, Parallel, and Cluster Computing
    CS_OS = "cs.OS"      # Operating Systems
    CS_AR = "cs.AR"      # Hardware Architecture


class OutputMode(Enum):
    """Output mode for paper fetching."""

    TITLE_ONLY = "title_only"
    TITLE_ABSTRACT = "title_abstract"


class MatchType(Enum):
    """Type of match for filtered papers."""

    SIMILARITY = "similarity"  # Matched by semantic similarity
    KEYWORD = "keyword"  # Matched by keyword in title/abstract
    AUTHOR = "author"  # Matched by followed author
    KEYWORD_AUTHOR = "keyword_author"  # Matched by both keyword and author


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
    keywords: list[str]  # Specific keywords to match (hard matching in title/abstract)
    past_papers: list[PastPaper]  # Papers user found interesting
    categories: list[str]  # arXiv categories to fetch (e.g., cs.LG, cs.AI)
    followed_authors: Optional[list[str]] = None  # Authors to follow (hard matching)
    name: str = "default"  # Profile identifier


@dataclass
class FilteredPaper:
    """Paper with its similarity score and match information."""

    paper: Paper
    similarity_score: float
    match_type: MatchType = MatchType.SIMILARITY
    matched_keywords: Optional[list[str]] = None
    matched_authors: Optional[list[str]] = None

    def __str__(self) -> str:
        if self.match_type == MatchType.SIMILARITY:
            return f"[{self.similarity_score:.3f}] {self.paper}"
        elif self.match_type == MatchType.KEYWORD:
            return f"[KEYWORD] {self.paper}"
        elif self.match_type == MatchType.AUTHOR:
            return f"[AUTHOR] {self.paper}"
        else:  # KEYWORD_AUTHOR
            return f"[KW+AUTH] {self.paper}"


