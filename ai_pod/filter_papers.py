"""Filter papers by semantic similarity using SPECTER2 embeddings."""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ai_pod.models import (
    ArxivCategory,
    FilteredPaper,
    OutputMode,
    Paper,
    PastPaper,
    UserProfile,
)
from ai_pod.utils.cache_utils import dict_to_paper, get_cache_path, load_cache
from ai_pod.utils.embedding_cache import (
    load_paper_embeddings,
    load_profile_embeddings,
    save_paper_embeddings,
    save_profile_embeddings,
)


# Global model cache to avoid reloading
_MODEL_CACHE: dict = {}


def get_device() -> str:
    """Auto-detect the best available device.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_specter_model(device: Optional[str] = None):
    """Load SPECTER2 model with proximity adapter.

    Args:
        device: Device to load model on. If None, auto-detects best available device.

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = get_device()
    
    cache_key = f"specter2_{device}"
    if cache_key in _MODEL_CACHE:
        logger.debug("Using cached SPECTER2 model")
        return _MODEL_CACHE[cache_key]

    logger.info("Loading SPECTER2 model (this may take a moment)...")

    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

    # Load the proximity adapter for finding similar papers
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2_proximity", set_active=True)

    model = model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    logger.info(f"SPECTER2 model loaded on {device}")

    return model, tokenizer


def generate_paper_embeddings(
    papers: list[Paper],
    tokenizer,
    model,
    batch_size: int = 32,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, list[float]]:
    """Generate embeddings for a list of papers.

    Args:
        papers: List of Paper objects.
        tokenizer: SPECTER2 tokenizer.
        model: SPECTER2 model.
        batch_size: Batch size for processing.
        device: Device to use. If None, auto-detects best available device.
        use_cache: Whether to use/update embedding cache.

    Returns:
        Dictionary mapping arxiv_id to embedding vector.
    """
    import torch

    if device is None:
        device = get_device()

    # Load cached embeddings
    cached_embeddings = load_paper_embeddings() if use_cache else {}
    if cached_embeddings is None:
        cached_embeddings = {}

    # Find papers that need embeddings
    papers_to_embed = [p for p in papers if p.arxiv_id not in cached_embeddings]

    if not papers_to_embed:
        logger.info(f"Using cached embeddings for all {len(papers)} papers")
        return {p.arxiv_id: cached_embeddings[p.arxiv_id] for p in papers}

    logger.info(f"Generating embeddings for {len(papers_to_embed)} papers ({len(papers) - len(papers_to_embed)} cached)")

    new_embeddings = {}

    # Process in batches
    for i in range(0, len(papers_to_embed), batch_size):
        batch = papers_to_embed[i : i + batch_size]

        # Format: title + [SEP] + abstract
        texts = []
        for paper in batch:
            if paper.abstract:
                text = f"{paper.title}{tokenizer.sep_token}{paper.abstract}"
            else:
                text = paper.title
            texts.append(text)

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Store embeddings
        for paper, embedding in zip(batch, embeddings):
            new_embeddings[paper.arxiv_id] = embedding.tolist()

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(papers_to_embed):
            logger.debug(f"Processed {min(i + batch_size, len(papers_to_embed))}/{len(papers_to_embed)} papers")

    # Save new embeddings to cache
    if use_cache and new_embeddings:
        save_paper_embeddings(new_embeddings)

    # Combine cached and new embeddings
    result = {p.arxiv_id: cached_embeddings.get(p.arxiv_id) or new_embeddings.get(p.arxiv_id) for p in papers}
    return result


def generate_profile_embeddings(
    profile: UserProfile,
    tokenizer,
    model,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> dict[str, list[list[float]]]:
    """Generate embeddings for a user profile.

    Args:
        profile: User profile with topics, keywords, and past papers.
        tokenizer: SPECTER2 tokenizer.
        model: SPECTER2 model.
        device: Device to use. If None, auto-detects best available device.
        use_cache: Whether to use/update embedding cache.

    Returns:
        Dictionary with 'topics', 'keywords', 'papers' keys mapping to lists of embeddings.
    """
    import torch

    if device is None:
        device = get_device()

    # Try loading from cache
    if use_cache:
        cached = load_profile_embeddings(profile)
        if cached is not None:
            logger.info(f"Using cached embeddings for profile '{profile.name}'")
            return cached

    logger.info(f"Generating embeddings for profile '{profile.name}'")

    def embed_texts(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings.tolist()

    # Generate embeddings for each category
    result = {
        "topics": embed_texts(profile.topics),
        "keywords": embed_texts(profile.keywords),
        "papers": [],
    }

    # Generate embeddings for past papers (format: title + [SEP] + abstract)
    if profile.past_papers:
        paper_texts = []
        for paper in profile.past_papers:
            if paper.abstract:
                text = f"{paper.title}{tokenizer.sep_token}{paper.abstract}"
            else:
                text = paper.title
            paper_texts.append(text)
        result["papers"] = embed_texts(paper_texts)

    # Save to cache
    if use_cache:
        save_profile_embeddings(profile, result)

    return result


def compute_similarity_scores(
    paper_embeddings: dict[str, list[float]],
    profile_embeddings: dict[str, list[list[float]]],
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Compute weighted similarity scores between papers and profile.

    Args:
        paper_embeddings: Dictionary mapping arxiv_id to embedding.
        profile_embeddings: Dictionary with 'topics', 'keywords', 'papers' embeddings.
        weights: Optional weight dictionary. Defaults to topics=0.4, keywords=0.3, papers=0.3.

    Returns:
        Dictionary mapping arxiv_id to similarity score.
    """
    if weights is None:
        weights = {"topics": 0.4, "keywords": 0.3, "papers": 0.3}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Convert profile embeddings to numpy arrays
    topics_emb = np.array(profile_embeddings.get("topics", [])) if profile_embeddings.get("topics") else None
    keywords_emb = np.array(profile_embeddings.get("keywords", [])) if profile_embeddings.get("keywords") else None
    papers_emb = np.array(profile_embeddings.get("papers", [])) if profile_embeddings.get("papers") else None

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def max_similarity_to_set(paper_emb: np.ndarray, set_emb: np.ndarray) -> float:
        """Compute maximum cosine similarity between a paper and a set of embeddings."""
        if set_emb is None or len(set_emb) == 0:
            return 0.0
        similarities = [cosine_similarity(paper_emb, s) for s in set_emb]
        return max(similarities)

    scores = {}
    for arxiv_id, emb in paper_embeddings.items():
        paper_emb = np.array(emb)

        # Compute similarity to each category
        topic_sim = max_similarity_to_set(paper_emb, topics_emb) if topics_emb is not None else 0.0
        keyword_sim = max_similarity_to_set(paper_emb, keywords_emb) if keywords_emb is not None else 0.0
        paper_sim = max_similarity_to_set(paper_emb, papers_emb) if papers_emb is not None else 0.0

        # Weighted average
        score = (
            weights.get("topics", 0) * topic_sim
            + weights.get("keywords", 0) * keyword_sim
            + weights.get("papers", 0) * paper_sim
        )

        # Adjust weights if some categories are empty
        active_weight = 0.0
        if topics_emb is not None and len(topics_emb) > 0:
            active_weight += weights.get("topics", 0)
        if keywords_emb is not None and len(keywords_emb) > 0:
            active_weight += weights.get("keywords", 0)
        if papers_emb is not None and len(papers_emb) > 0:
            active_weight += weights.get("papers", 0)

        if active_weight > 0:
            score = score / active_weight

        scores[arxiv_id] = score

    return scores


def filter_papers(
    papers: list[Paper],
    profile: UserProfile,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    use_cache: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> list[FilteredPaper]:
    """Filter papers by semantic similarity to user profile.

    Args:
        papers: List of papers to filter.
        profile: User profile with interests.
        threshold: Minimum similarity score to include (0.0 to 1.0).
        top_k: If set, return top-k papers instead of using threshold.
        use_cache: Whether to use embedding cache.
        batch_size: Batch size for embedding generation.
        device: Device for model. If None, auto-detects best available device.

    Returns:
        List of FilteredPaper objects sorted by similarity score (descending).
    """
    if not papers:
        logger.warning("No papers to filter")
        return []

    if not profile.topics and not profile.keywords and not profile.past_papers:
        logger.warning("Empty profile - no topics, keywords, or past papers")
        return []

    if device is None:
        device = get_device()

    # Load model
    model, tokenizer = get_specter_model(device)

    # Generate embeddings
    paper_embeddings = generate_paper_embeddings(
        papers, tokenizer, model, batch_size, device, use_cache
    )
    profile_embeddings = generate_profile_embeddings(
        profile, tokenizer, model, device, use_cache
    )

    # Compute similarity scores
    scores = compute_similarity_scores(paper_embeddings, profile_embeddings)

    # Create FilteredPaper objects
    filtered = [
        FilteredPaper(paper=p, similarity_score=scores.get(p.arxiv_id, 0.0))
        for p in papers
    ]

    # Sort by score descending
    filtered.sort(key=lambda x: x.similarity_score, reverse=True)

    # Apply threshold or top_k
    if top_k is not None:
        filtered = filtered[:top_k]
    else:
        filtered = [f for f in filtered if f.similarity_score >= threshold]

    logger.info(f"Filtered {len(papers)} papers to {len(filtered)} papers")

    return filtered


def load_user_profile(profile_path: str | Path) -> UserProfile:
    """Load a user profile from a JSON file.

    Args:
        profile_path: Path to the profile JSON file.

    Returns:
        UserProfile object.
    """
    path = Path(profile_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    past_papers = [
        PastPaper(title=p["title"], abstract=p.get("abstract"))
        for p in data.get("past_papers", [])
    ]

    return UserProfile(
        name=data.get("name", "default"),
        topics=data.get("topics", []),
        keywords=data.get("keywords", []),
        past_papers=past_papers,
        preferred_authors=data.get("preferred_authors"),
    )


def save_user_profile(profile: UserProfile, profile_path: str | Path) -> None:
    """Save a user profile to a JSON file.

    Args:
        profile: UserProfile object to save.
        profile_path: Path to save the profile to.
    """
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": profile.name,
        "topics": profile.topics,
        "keywords": profile.keywords,
        "past_papers": [
            {"title": p.title, "abstract": p.abstract} for p in profile.past_papers
        ],
        "preferred_authors": profile.preferred_authors or [],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved profile to {path}")


def main():
    """CLI entry point for paper filtering."""
    parser = argparse.ArgumentParser(description="Filter papers by semantic similarity")

    parser.add_argument(
        "-p", "--profile",
        type=str,
        required=True,
        help="Path to user profile JSON file",
    )

    # Paper source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--papers-cache",
        type=str,
        help="Path to cached papers JSON file",
    )
    source_group.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh papers from arXiv",
    )

    # Fetch options (when using --fetch)
    category_choices = [cat.value for cat in ArxivCategory]
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=category_choices,
        default=["cs.LG", "cs.AI", "cs.CL"],
        help="arXiv categories to search (default: cs.LG cs.AI cs.CL)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=7,
        help="Number of days back to search (default: 7)",
    )

    # Filtering options
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Minimum similarity threshold (default: 0.5)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=None,
        help="Return top-k papers instead of using threshold",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )

    args = parser.parse_args()

    # Load user profile
    logger.info(f"Loading profile from {args.profile}")
    profile = load_user_profile(args.profile)
    logger.info(f"Profile '{profile.name}': {len(profile.topics)} topics, {len(profile.keywords)} keywords, {len(profile.past_papers)} past papers")

    # Get papers
    if args.fetch:
        from ai_pod.get_papers import get_papers

        categories = [ArxivCategory(cat) for cat in args.categories]
        logger.info(f"Fetching papers from {args.categories} (last {args.days} days)...")
        papers = get_papers(
            categories=categories,
            days=args.days,
            max_results=-1,  # Get all papers
            mode=OutputMode.TITLE_ABSTRACT,
        )
    else:
        # Load from cache file
        logger.info(f"Loading papers from {args.papers_cache}")
        with open(args.papers_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        papers = [dict_to_paper(p) for p in data.get("papers", [])]

    if not papers:
        print("No papers found.")
        return

    logger.info(f"Loaded {len(papers)} papers")

    # Filter papers
    filtered = filter_papers(
        papers=papers,
        profile=profile,
        threshold=args.threshold,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
    )

    # Output results
    print("-" * 60)
    if args.top_k:
        print(f"Top {len(filtered)} papers for profile '{profile.name}':\n")
    else:
        print(f"Found {len(filtered)} papers above threshold {args.threshold}:\n")

    for i, fp in enumerate(filtered, 1):
        print(f"{i}. [Score: {fp.similarity_score:.3f}] {fp.paper}")
        if fp.paper.abstract:
            abstract = fp.paper.abstract[:200] + "..." if len(fp.paper.abstract) > 200 else fp.paper.abstract
            print(f"   Abstract: {abstract}")
        print()


if __name__ == "__main__":
    main()
