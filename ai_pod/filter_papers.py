"""Filter papers by semantic similarity using SPECTER2 embeddings."""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ai_pod.models import (
    ArxivCategory,
    FilteredPaper,
    MatchType,
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


def match_papers_by_keywords(
    papers: list[Paper],
    keywords: list[str],
) -> dict[str, list[str]]:
    """Match papers by keyword presence in title or abstract.

    Args:
        papers: List of papers to match against.
        keywords: List of keywords to search for.

    Returns:
        Dictionary mapping arxiv_id to list of matched keywords.
    """
    if not keywords:
        return {}

    matches: dict[str, list[str]] = {}

    for paper in papers:
        # Combine title and abstract for searching
        text = paper.title.lower()
        if paper.abstract:
            text += " " + paper.abstract.lower()

        matched_keywords = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Case-insensitive substring match
            if keyword_lower in text:
                matched_keywords.append(keyword)

        if matched_keywords:
            matches[paper.arxiv_id] = matched_keywords

    return matches


def normalize_author_name(name: str) -> str:
    """Normalize author name for matching.

    - Remove affiliations in parentheses
    - Lowercase
    - Normalize whitespace

    Args:
        name: Author name to normalize.

    Returns:
        Normalized author name.
    """
    # Remove affiliations in parentheses (e.g., "Song Han (MIT)" -> "Song Han")
    name = re.sub(r'\s*\([^)]*\)\s*', '', name)
    # Lowercase and normalize whitespace
    name = ' '.join(name.lower().split())
    return name


def match_papers_by_authors(
    papers: list[Paper],
    followed_authors: list[str],
) -> dict[str, list[str]]:
    """Match papers by author presence.

    Args:
        papers: List of papers to match against.
        followed_authors: List of followed author names.

    Returns:
        Dictionary mapping arxiv_id to list of matched author names.
    """
    if not followed_authors:
        return {}

    # Normalize followed author names (skip empty strings)
    normalized_followed = {}
    for author in followed_authors:
        author = author.strip()
        if author:  # Skip empty strings
            normalized_followed[normalize_author_name(author)] = author

    if not normalized_followed:
        return {}

    matches: dict[str, list[str]] = {}

    for paper in papers:
        matched_authors = []

        for paper_author in paper.authors:
            normalized_paper_author = normalize_author_name(paper_author.name)

            # Check if any followed author matches
            for normalized_followed_author, original_name in normalized_followed.items():
                # Check if the normalized names match
                if normalized_paper_author == normalized_followed_author:
                    matched_authors.append(original_name)
                # Also check if one contains the other (for partial name matching)
                elif (normalized_followed_author in normalized_paper_author or
                      normalized_paper_author in normalized_followed_author):
                    matched_authors.append(original_name)

        if matched_authors:
            # Remove duplicates while preserving order
            seen = set()
            unique_matched = []
            for name in matched_authors:
                if name not in seen:
                    seen.add(name)
                    unique_matched.append(name)
            matches[paper.arxiv_id] = unique_matched

    return matches


def filter_papers(
    papers: list[Paper],
    profile: UserProfile,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    use_cache: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> list[FilteredPaper]:
    """Filter papers by hard matching and semantic similarity.

    Filtering flow:
    1. Hard match by keywords (case-insensitive substring in title/abstract)
    2. Hard match by followed_authors (author name matching)
    3. Deduplicate hard matches
    4. For remaining papers: compute similarity scores (topics + past_papers only)
    5. Combine: hard matches first (score=1.0), then similarity-based

    Args:
        papers: List of papers to filter.
        profile: User profile with interests.
        threshold: Minimum similarity score for non-hard-matched papers (0.0 to 1.0).
        top_k: If set, return top-k papers instead of using threshold.
        use_cache: Whether to use embedding cache.
        batch_size: Batch size for embedding generation.
        device: Device for model. If None, auto-detects best available device.

    Returns:
        List of FilteredPaper objects sorted by match type and score (descending).
    """
    if not papers:
        logger.warning("No papers to filter")
        return []

    # Check if profile has any filtering criteria
    has_keywords = bool(profile.keywords)
    has_followed_authors = bool(profile.followed_authors and any(a.strip() for a in profile.followed_authors))
    has_topics = bool(profile.topics)
    has_past_papers = bool(profile.past_papers)

    if not has_keywords and not has_followed_authors and not has_topics and not has_past_papers:
        logger.warning("Empty profile - no keywords, followed_authors, topics, or past papers")
        return []

    # Phase 1: Hard matching (keywords + authors)
    keyword_matches = match_papers_by_keywords(papers, profile.keywords) if has_keywords else {}
    author_matches = match_papers_by_authors(papers, profile.followed_authors or []) if has_followed_authors else {}

    logger.info(f"Hard matches: {len(keyword_matches)} by keywords, {len(author_matches)} by authors")

    # Build hard-matched papers
    hard_matched: dict[str, FilteredPaper] = {}
    paper_lookup = {p.arxiv_id: p for p in papers}

    for arxiv_id, matched_keywords in keyword_matches.items():
        paper = paper_lookup[arxiv_id]
        if arxiv_id in author_matches:
            # Matched by both keyword and author
            hard_matched[arxiv_id] = FilteredPaper(
                paper=paper,
                similarity_score=1.0,
                match_type=MatchType.KEYWORD_AUTHOR,
                matched_keywords=matched_keywords,
                matched_authors=author_matches[arxiv_id],
            )
        else:
            # Matched by keyword only
            hard_matched[arxiv_id] = FilteredPaper(
                paper=paper,
                similarity_score=1.0,
                match_type=MatchType.KEYWORD,
                matched_keywords=matched_keywords,
            )

    for arxiv_id, matched_authors in author_matches.items():
        if arxiv_id not in hard_matched:
            # Matched by author only
            paper = paper_lookup[arxiv_id]
            hard_matched[arxiv_id] = FilteredPaper(
                paper=paper,
                similarity_score=1.0,
                match_type=MatchType.AUTHOR,
                matched_authors=matched_authors,
            )

    # Phase 2: Determine if we need semantic similarity
    # We need similarity if:
    # 1. Hard matches exceed top_k (need to rank hard matches)
    # 2. There are remaining papers and we have topics/past_papers for similarity
    need_similarity_for_hard_matches = top_k is not None and len(hard_matched) > top_k
    remaining_papers = [p for p in papers if p.arxiv_id not in hard_matched]
    need_similarity_for_remaining = remaining_papers and (has_topics or has_past_papers)

    similarity_filtered: list[FilteredPaper] = []
    hard_match_scores: dict[str, float] = {}

    if (need_similarity_for_hard_matches or need_similarity_for_remaining) and (has_topics or has_past_papers):
        if device is None:
            device = get_device()

        # Load model
        model, tokenizer = get_specter_model(device)

        # Determine which papers need embeddings
        papers_for_embeddings = []
        if need_similarity_for_hard_matches:
            papers_for_embeddings.extend([paper_lookup[aid] for aid in hard_matched.keys()])
        if need_similarity_for_remaining:
            papers_for_embeddings.extend(remaining_papers)

        # Generate embeddings
        paper_embeddings = generate_paper_embeddings(
            papers_for_embeddings, tokenizer, model, batch_size, device, use_cache
        )
        profile_embeddings = generate_profile_embeddings(
            profile, tokenizer, model, device, use_cache
        )

        # Use only topics and past_papers for similarity (no keywords - they're for hard matching)
        weights = {"topics": 0.5, "keywords": 0.0, "papers": 0.5}
        scores = compute_similarity_scores(paper_embeddings, profile_embeddings, weights)

        # Store scores for hard matches if needed
        if need_similarity_for_hard_matches:
            for arxiv_id in hard_matched.keys():
                hard_match_scores[arxiv_id] = scores.get(arxiv_id, 0.0)

        # Create FilteredPaper objects for remaining papers
        if need_similarity_for_remaining:
            similarity_filtered = [
                FilteredPaper(
                    paper=p,
                    similarity_score=scores.get(p.arxiv_id, 0.0),
                    match_type=MatchType.SIMILARITY,
                )
                for p in remaining_papers
            ]

            # Sort by score descending
            similarity_filtered.sort(key=lambda x: x.similarity_score, reverse=True)

            # Apply threshold to similarity-based matches
            similarity_filtered = [f for f in similarity_filtered if f.similarity_score >= threshold]

    # Phase 3: Combine results
    hard_matched_list = list(hard_matched.values())

    # If hard matches exceed top_k, sort by semantic similarity to pick the most relevant
    if need_similarity_for_hard_matches and hard_match_scores:
        # Update similarity scores for hard matches (keep match_type, but use score for ranking)
        for fp in hard_matched_list:
            fp.similarity_score = hard_match_scores.get(fp.paper.arxiv_id, 0.0)
        # Sort by similarity score descending
        hard_matched_list.sort(key=lambda x: x.similarity_score, reverse=True)
        logger.info(f"Hard matches ({len(hard_matched_list)}) exceed top_k ({top_k}), using semantic similarity to rank")
    else:
        # Sort hard matches: KEYWORD_AUTHOR first, then KEYWORD, then AUTHOR
        match_type_order = {
            MatchType.KEYWORD_AUTHOR: 0,
            MatchType.KEYWORD: 1,
            MatchType.AUTHOR: 2,
        }
        hard_matched_list.sort(key=lambda x: match_type_order.get(x.match_type, 3))

    combined = hard_matched_list + similarity_filtered

    # Apply top_k if specified (after combining)
    if top_k is not None:
        combined = combined[:top_k]

    logger.info(
        f"Filtered {len(papers)} papers to {len(combined)} papers "
        f"({min(len(hard_matched_list), len(combined))} hard matches, {max(0, len(combined) - len(hard_matched_list))} by similarity)"
    )

    return combined


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
        PastPaper(title=p["title"], abstract=p.get("abstract"), arxiv_id=p.get("arxiv_id"))
        for p in data.get("past_papers", [])
    ]

    # Support both followed_authors (new) and preferred_authors (legacy) for backward compatibility
    followed_authors = data.get("followed_authors") or data.get("preferred_authors")

    return UserProfile(
        name=data.get("name", "default"),
        topics=data.get("topics", []),
        keywords=data.get("keywords", []),
        past_papers=past_papers,
        categories=data.get("categories", ["cs.LG", "cs.AI", "cs.CL"]),
        followed_authors=followed_authors,
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
            {"title": p.title, "abstract": p.abstract, "arxiv_id": p.arxiv_id}
            for p in profile.past_papers
        ],
        "categories": profile.categories,
        "followed_authors": profile.followed_authors or [],
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
        # Format match type label
        if fp.match_type == MatchType.KEYWORD_AUTHOR:
            match_label = "[KW+AUTH]"
            extra_info = f" (keywords: {', '.join(fp.matched_keywords or [])}, authors: {', '.join(fp.matched_authors or [])})"
        elif fp.match_type == MatchType.KEYWORD:
            match_label = "[KEYWORD]"
            extra_info = f" (matched: {', '.join(fp.matched_keywords or [])})"
        elif fp.match_type == MatchType.AUTHOR:
            match_label = "[AUTHOR]"
            extra_info = f" (matched: {', '.join(fp.matched_authors or [])})"
        else:
            match_label = f"[{fp.similarity_score:.3f}]"
            extra_info = ""

        print(f"{i}. {match_label} {fp.paper}{extra_info}")
        if fp.paper.abstract:
            abstract = fp.paper.abstract[:200] + "..." if len(fp.paper.abstract) > 200 else fp.paper.abstract
            print(f"   Abstract: {abstract}")
        print()


if __name__ == "__main__":
    main()
