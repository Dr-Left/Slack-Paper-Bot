"""LLM summary generation for paper digests."""

from typing import Optional

import numpy as np
import re
from loguru import logger

from ai_pod.filter_papers import get_specter_model
from ai_pod.models import FilteredPaper
from ai_pod.utils.embedding_cache import load_paper_embeddings


def find_similar_past_paper(
    paper: FilteredPaper,
    previous_papers: list[dict],
    threshold: float = 0.9,
) -> Optional[tuple[dict, float]]:
    """Find the most similar past paper using SPECTER2 embeddings.

    Args:
        paper: The current paper to compare.
        previous_papers: List of previously posted papers (with title, abstract).
        threshold: Minimum similarity threshold (default: 0.9).

    Returns:
        Tuple of (past_paper_dict, similarity) if similarity > threshold, else None.
    """
    if not previous_papers:
        return None

    # Filter to papers with titles
    past_with_titles = [p for p in previous_papers if p.get("title")]
    if not past_with_titles:
        return None

    # Load model and cached embeddings
    model, tokenizer = get_specter_model()
    cached_embeddings = load_paper_embeddings() or {}

    # Get embedding for current paper (should already be cached from filtering)
    current_emb = cached_embeddings.get(paper.paper.arxiv_id)
    if current_emb is None:
        logger.debug(f"No cached embedding for {paper.paper.arxiv_id}")
        return None

    current_emb = np.array(current_emb)

    # Find past papers with cached embeddings
    best_match = None
    best_similarity = 0.0

    for past_paper in past_with_titles:
        past_id = past_paper.get("arxiv_id")
        if not past_id or past_id == paper.paper.arxiv_id:
            continue

        past_emb = cached_embeddings.get(past_id)
        if past_emb is None:
            continue

        past_emb = np.array(past_emb)

        # Compute cosine similarity
        norm_current = np.linalg.norm(current_emb)
        norm_past = np.linalg.norm(past_emb)
        if norm_current == 0 or norm_past == 0:
            continue

        similarity = float(np.dot(current_emb, past_emb) / (norm_current * norm_past))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = past_paper

    if best_match and best_similarity > threshold:
        logger.debug(f"Found similar past paper: {best_match.get('title', 'Unknown')} ({best_similarity:.2f})")
        return (best_match, best_similarity)
    else:
        logger.debug(f"No similar past paper found for {paper.paper.arxiv_id}")
        return None


def generate_paper_summary(
    papers: list[FilteredPaper],
    previous_papers: list[dict],
    openai_api_key: str,
) -> Optional[str]:
    """Generate a summary of recommended papers using OpenAI gpt-4o.

    Args:
        papers: List of papers being recommended today.
        previous_papers: List of previously posted papers (for contrast).
        openai_api_key: OpenAI API key.

    Returns:
        Summary text (<200 words), or None if generation fails.
    """
    if not papers:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)

        # Build prompt with paired current/contrastive papers
        paper_lines = []
        contrast_pairs: list[Optional[dict]] = []
        for i, fp in enumerate(papers, 1):
            title = fp.paper.title
            abstract_snippet = (
                fp.paper.abstract[:150] + "..."
                if fp.paper.abstract and len(fp.paper.abstract) > 150
                else (fp.paper.abstract or "")
            )

            # Start line with current paper info
            line = f'{i}. CURRENT: "{title}" - {abstract_snippet}'

            # Check for similar past (contrastive) paper
            similar = find_similar_past_paper(fp, previous_papers, threshold=0.8)
            if similar:
                past_paper, sim_score = similar
                contrast_title = past_paper.get("title", "Unknown")
                contrast_abstract = past_paper.get("abstract") or ""
                if contrast_abstract and len(contrast_abstract) > 150:
                    contrast_abstract = contrast_abstract[:150] + "..."

                line += (
                    f'\n   CONTRAST: "{contrast_title}"'
                    f" (similarity {sim_score:.2f}) - {contrast_abstract}"
                )
                contrast_pairs.append(
                    {
                        "title": contrast_title,
                        "arxiv_id": past_paper.get("arxiv_id"),
                    }
                )
            else:
                contrast_pairs.append(None)

            paper_lines.append(line)

        prompt = (
            "You are preparing a Slack digest summary.\n"
            "Each item below is a recommended paper, optionally paired with a similar past paper "
            "marked as CONTRAST.\n\n"
            "PAPERS (ONE PAIR PER LINE):\n"
            + "\n\n".join(paper_lines)
            + "\n\n"
            "Write a concise summary under 200 words as bullet points:\n"
            "- Use one bullet per CURRENT/CONTRAST pair in order.\n"
            "- For each bullet, briefly explain why the CURRENT paper is interesting and how it contrasts with its CONTRAST paper (if present).\n"
            "- If there is no CONTRAST paper for a CURRENT paper, still include a bullet but just describe the CURRENT paper.\n"
            "- Keep bullets tight and focused for a Slack message."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize ML paper recommendations for a Slack digest. "
                        "Respond ONLY with markdown bullet points, each describing one CURRENT paper "
                        "and, when provided, its CONTRAST paper. Stay under 300 words total."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # Keep this modest; we also hard-truncate during post-processing below.
            max_tokens=500,
            temperature=0.7,
        )

        summary = response.choices[0].message.content

        def truncate_words_preserving_slack_links(text: str, max_words: int) -> str:
            """Truncate to max_words, treating Slack link markup (<url|label>) as a single token."""
            if max_words <= 0:
                return ""
            # Keep <...> intact, otherwise split on whitespace.
            tokens = re.findall(r"<[^>\n]+>|\S+", text)
            if len(tokens) <= max_words:
                return text.strip()
            return (" ".join(tokens[:max_words]).rstrip() + " ...").strip()

        def strip_simple_markdown(text: str) -> str:
            # Remove lightweight emphasis markers that can render oddly in Slack
            for token in ("**", "*", "_", "`"):
                text = text.replace(token, "")
            return text.strip()

        # Post-process to ensure contrast links/titles are explicitly mentioned
        # and normalize bullets to a Slack-friendly format.
        lines = summary.splitlines()
        processed: list[str] = []
        contrast_idx = 0
        for line in lines:
            stripped = line.strip()
            is_bullet = stripped.startswith(("-", "*", "•"))

            # Normalize bullet content
            content = stripped.lstrip("-*•").lstrip() if is_bullet else stripped

            # Append contrast link when available
            if is_bullet:
                contrast_raw = contrast_pairs[contrast_idx] if contrast_idx < len(contrast_pairs) else None
                contrast = contrast_raw if isinstance(contrast_raw, dict) else None
                if contrast:
                    arxiv_id = contrast.get("arxiv_id")
                    contrast_title = contrast.get("title")
                    if arxiv_id and contrast_title:
                        link = f"<https://arxiv.org/abs/{arxiv_id}|{contrast_title}>"
                        content = f"{content} (contrast: {link})"
                contrast_idx += 1

            content = strip_simple_markdown(content)

            if is_bullet:
                # Hard cap per-paper (per-bullet) length to keep Slack blocks valid.
                content = truncate_words_preserving_slack_links(content, max_words=100)
                processed.append(f"• {content}")
            elif content:
                processed.append(content)

        return "\n".join(processed)

    except Exception as e:
        logger.warning(f"Failed to generate paper summary: {e}")
        return None
