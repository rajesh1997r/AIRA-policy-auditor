"""
Clause segmenter for policy documents.

Splits raw policy text into a list of clause dicts with deterministic IDs.
"""

import re
import hashlib
from typing import List, Dict


def _make_clause_id(doc_prefix: str, text: str) -> str:
    """Deterministic ID: prefix + first 6 chars of MD5 hash of first 40 chars."""
    snippet = text[:40].encode("utf-8")
    return f"{doc_prefix}_{hashlib.md5(snippet).hexdigest()[:6].upper()}"


def _detect_heading(line: str) -> bool:
    """Returns True if line looks like a section heading."""
    stripped = line.strip()
    if not stripped:
        return False
    # ALL CAPS line, or line ending with ':'
    if stripped.endswith(":") and len(stripped) < 80:
        return True
    if stripped == stripped.upper() and len(stripped) > 3 and not stripped[0].isdigit():
        return True
    return False


def _split_into_candidates(text: str) -> List[str]:
    """
    Split text into candidate clause strings.

    Splits on:
    - Blank lines (paragraph breaks)
    - Numbered list items: '1.', '2)', 'a.', 'b)'
    - Bullet list items: '•', '-', '*' at start of line
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank lines first
    paragraphs = re.split(r"\n\s*\n", text)

    candidates = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if paragraph contains numbered/bulleted list items
        lines = para.split("\n")
        if len(lines) == 1:
            candidates.append(para)
        else:
            # Check if any line starts with a list marker
            list_pattern = re.compile(
                r"^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[•\-\*])\s+"
            )
            has_list = any(list_pattern.match(l) for l in lines)

            if has_list:
                # Split on list item boundaries
                buffer = []
                for line in lines:
                    if list_pattern.match(line) and buffer:
                        candidates.append(" ".join(buffer))
                        buffer = [line.strip()]
                    else:
                        buffer.append(line.strip())
                if buffer:
                    candidates.append(" ".join(buffer))
            else:
                # Treat whole paragraph as one clause
                candidates.append(" ".join(l.strip() for l in lines))

    return candidates


def segment_policy(text: str, doc_prefix: str = "DOC") -> List[Dict]:
    """
    Segment a policy document into a list of clause dicts.

    Args:
        text: Raw policy document as a string.
        doc_prefix: Short prefix for clause IDs (e.g. 'NEU', 'MIT').

    Returns:
        List of dicts with keys: id, text, source_doc, section.
        Clauses with fewer than 15 whitespace-separated tokens are dropped.
    """
    candidates = _split_into_candidates(text)

    clauses = []
    current_section = "GENERAL"

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        if _detect_heading(candidate):
            current_section = candidate.strip().rstrip(":").strip()
            continue

        # Filter fragments: require at least 15 tokens
        tokens = candidate.split()
        if len(tokens) < 15:
            continue

        clause_id = _make_clause_id(doc_prefix, candidate)
        clauses.append(
            {
                "id": clause_id,
                "text": candidate,
                "source_doc": doc_prefix,
                "section": current_section,
            }
        )

    return clauses


if __name__ == "__main__":
    sample = """PERMITTED USES

Students may use AI tools for research purposes with appropriate attribution. Any AI-assisted work must be disclosed to the instructor in writing before submission.

AI-assisted brainstorming is allowed and encouraged as a starting point for original thinking. Students remain responsible for all content in the final submission.

PROHIBITED USES

Submitting AI-generated text as your own work is strictly prohibited and constitutes academic misconduct. Violations will be reported to the Academic Integrity Board.

The use of AI tools to generate code, essays, lab reports, or any other graded work is forbidden unless explicitly authorized by the course instructor in writing.

DISCLOSURE REQUIREMENTS

All students must disclose any use of AI tools in an acknowledgment section at the end of their submission. Failure to disclose constitutes a separate violation of this policy.

Instructors are not required to accept AI-assisted work and may set more restrictive policies for their individual courses."""

    clauses = segment_policy(sample, "NEU")
    for c in clauses:
        print(f"[{c['id']}] ({c['section']}) {c['text'][:80]}...")
    print(f"\nTotal clauses: {len(clauses)}")
