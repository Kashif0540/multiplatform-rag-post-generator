"""
MultiPlatform RAG Post Generator — Streamlit app (Phase 2)

Wraps the exact ingestion -> chunking -> local-embedding -> FAISS -> retrieval
-> platform-profiled Groq generation pipeline validated in Phase 1's Colab
notebook (multiplatform_rag_post_generator_test.ipynb) in an upload/URL +
platform-picker interface. No pipeline logic is rewritten here — only the
key-resolution and file-reading glue changes to fit Streamlit
"""

import csv
import difflib
import html as html_lib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import trafilatura
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =============================================================================
# Config
# =============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

PLATFORM_PROFILES: Dict[str, Dict[str, Any]] = {
    "LinkedIn": {
        "tone": "professional, thought-leadership",
        "length_guidance": "roughly 150-300 words",
        "formatting": "short paragraphs separated by line breaks, easy to skim",
        "hashtag_count": "3-5 hashtags at the end",
        "emoji_guidance": "at most 1-2 emoji, used sparingly",
        "hard_char_limit": None,  # LinkedIn's real cap (~3000 chars) isn't the binding constraint here
    },
    "X": {
        "tone": "concise and punchy",
        "length_guidance": "as short as possible while making the point",
        "formatting": "no line breaks needed, single flowing block",
        "hashtag_count": "1-2 hashtags max",
        "emoji_guidance": "0-1 emoji, optional",
        "hard_char_limit": 280,  # includes hashtags
    },
    "Instagram": {
        "tone": "engaging, conversational, emoji-friendly",
        "length_guidance": "a strong hook as the very first line, then a short caption",
        "formatting": "caption first, then a separate hashtag block afterward",
        "hashtag_count": "8-15 hashtags in a separate block after the caption",
        "emoji_guidance": "emoji encouraged throughout, used naturally",
        "hard_char_limit": None,  # Instagram's real cap (~2200 chars) isn't the binding constraint here
    },
    "Facebook": {
        "tone": "conversational, community-oriented",
        "length_guidance": "longer than X, shorter than LinkedIn",
        "formatting": "should end with a question or prompt inviting comments",
        "hashtag_count": "0-2 hashtags, light use only",
        "emoji_guidance": "1-2 emoji, optional",
        "hard_char_limit": None,
    },
    "Threads": {
        "tone": "casual and personal, slightly warmer than X",
        "length_guidance": "roughly 500 characters as a soft cap",
        "formatting": "no line breaks needed, single flowing block",
        "hashtag_count": "0-2 hashtags",
        "emoji_guidance": "0-1 emoji, optional",
        "hard_char_limit": 500,  # soft cap treated as the flag threshold
    },
}

# =============================================================================
# Ingestion — files + URLs
# (fetch_url_text mirrors Phase 1 exactly; the PDF/TXT helpers are the
# Streamlit-file-object equivalents of Phase 1's extract_pdf_text)
# =============================================================================


def fetch_url_text(url: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a URL and extract its main article/body text with trafilatura.

    Returns (text, error) — exactly one is non-None. Never fabricates content.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGPostGenerator/1.0)"}

    try:
        response = requests.get(url, timeout=timeout, headers=headers)
    except requests.exceptions.Timeout:
        return None, f"Timed out after {timeout}s while fetching: {url}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to: {url}"
    except requests.exceptions.RequestException as exc:
        return None, f"Failed to fetch {url}: {exc}"

    if response.status_code != 200:
        return None, f"Received HTTP {response.status_code} for: {url}"

    extracted = trafilatura.extract(
        response.text, include_comments=False, include_tables=False, favor_precision=True
    )

    if not extracted or not extracted.strip():
        return None, f"No extractable article text found at: {url} (page may be JS-rendered or blocked scraping)"

    return extracted.strip(), None


def extract_text_from_pdf(file_obj, name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract text from an uploaded PDF file-like object using pypdf."""
    try:
        reader = PdfReader(file_obj)
    except Exception as exc:
        return None, f"Could not open PDF '{name}': {exc}"

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue  # skip unreadable pages rather than failing the whole doc

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        return None, f"No extractable text found in PDF: '{name}'"

    return full_text, None


def extract_text_from_txt(file_obj, name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract text from an uploaded .txt file-like object."""
    try:
        raw = file_obj.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    except Exception as exc:
        return None, f"Could not read text file '{name}': {exc}"

    text = text.strip()
    if not text:
        return None, f"File '{name}' is empty or has no extractable text."

    return text, None


def ingest_sources(uploaded_files, url_block: str) -> Tuple[Dict[str, str], List[str]]:
    """Ingest uploaded files + pasted URLs into a {source_name: text} dict.

    Returns (documents, errors). Never fabricates content for a failed source
    — failed sources are reported in `errors` and simply excluded.
    """
    documents: Dict[str, str] = {}
    errors: List[str] = []

    for uploaded_file in uploaded_files or []:
        name = uploaded_file.name
        suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

        if suffix == "txt":
            text, err = extract_text_from_txt(uploaded_file, name)
        elif suffix == "pdf":
            text, err = extract_text_from_pdf(uploaded_file, name)
        else:
            text, err = None, f"Unsupported file type for '{name}' (only .txt and .pdf are supported)."

        if err:
            errors.append(err)
        else:
            documents[name] = text

    urls = [u.strip() for u in (url_block or "").splitlines() if u.strip()]
    for url in urls:
        text, err = fetch_url_text(url)
        if err:
            errors.append(err)
        else:
            documents[url] = text

    return documents, errors


# =============================================================================
# Chunking (identical to Phase 1)
# =============================================================================


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """Split `text` into overlapping chunks, each tagged with its source."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    text_len = len(text)
    step = chunk_size - overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
            chunk_id += 1
        if end == text_len:
            break
        start += step

    return chunks


def chunk_documents(documents: Dict[str, str], chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """Chunk every document in a {source: text} dict and return one flat list."""
    all_chunks: List[Chunk] = []
    for source, text in documents.items():
        all_chunks.extend(chunk_text(text, source, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


# =============================================================================
# Embedding + FAISS (identical to Phase 1)
# =============================================================================


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class VectorStore:
    """Local FAISS-backed vector store over sentence-transformers embeddings."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_embedding)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "source": chunk.source, "score": float(score)})
        return results


# =============================================================================
# Platform-profiled prompt builder (identical core logic to Phase 1; extended
# with an optional tone_nudge — extra free-text style guidance from the user,
# layered on top of the platform profile, not a replacement for it)
# =============================================================================


def build_prompt(
    topic: str,
    platform: str,
    retrieved_chunks: List[Dict[str, Any]],
    tone_nudge: Optional[str] = None,
) -> str:
    if platform not in PLATFORM_PROFILES:
        raise ValueError(f"Unknown platform: {platform}")

    profile = PLATFORM_PROFILES[platform]
    context_block = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks)

    tone_nudge_block = ""
    if tone_nudge and tone_nudge.strip():
        tone_nudge_block = (
            f"\nADDITIONAL STYLE NOTE FROM THE USER (apply on top of the platform profile above, "
            f"without breaking its length/hashtag/formatting rules): {tone_nudge.strip()}\n"
        )

    prompt = f"""You are a social media copywriter. Write ONE social media post for the platform: {platform}.

TOPIC / ANGLE TO WRITE ABOUT:
{topic}

GROUNDING CONTEXT (retrieved from the user's own source material — use ONLY these facts, do not invent facts not present in it):
---
{context_block}
---

PLATFORM PROFILE FOR {platform} (follow this exactly):
- Tone: {profile['tone']}
- Length: {profile['length_guidance']}
- Formatting: {profile['formatting']}
- Hashtags: {profile['hashtag_count']}
- Emoji: {profile['emoji_guidance']}
{tone_nudge_block}
Respond with ONLY a single JSON object, no markdown code fences, no extra commentary, shaped exactly like this:
{{"post": "<the full post text, including any hashtags that belong inside the post body>", "hashtags": ["<hashtag1>", "<hashtag2>"], "platform": "{platform}", "char_count": <integer character count of the post field>}}
"""
    return prompt


# =============================================================================
# Groq generation (identical logic to Phase 1; client resolved via st.secrets
# instead of Colab's userdata/getpass, and held in a module-level variable
# that call_groq() reads — mirroring the notebook's global groq_client)
# =============================================================================

groq_client: Optional[Groq] = None  # set at app start once the key is resolved


def call_groq(prompt: str, model: str = GROQ_MODEL, temperature: float = 0.7) -> Dict[str, Any]:
    """Call Groq chat completions in JSON object mode and return the parsed dict."""
    if groq_client is None:
        raise RuntimeError("Groq client is not configured — GROQ_API_KEY is missing.")

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "auth" in message:
            raise RuntimeError("Groq authentication failed — double-check your GROQ_API_KEY.") from exc
        if "429" in message or "rate" in message:
            raise RuntimeError("Groq rate limit hit — wait a moment and try again.") from exc
        if "timeout" in message or "connection" in message:
            raise RuntimeError(f"Groq network error: {exc}") from exc
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    raw = response.choices[0].message.content or ""
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse Groq JSON response after cleanup: {exc}\nRaw response: {raw[:300]}")

    raise ValueError(f"Groq response was not valid JSON: {raw[:300]}")


# =============================================================================
# Full pipeline (identical to Phase 1)
# =============================================================================


def generate_posts(
    vector_store: VectorStore,
    topic: str,
    platforms: List[str],
    k: int = 3,
    tone_nudge: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve once for `topic`, then generate one platform-tailored post per entry in `platforms`."""
    retrieved = vector_store.search(topic, k=k)
    if not retrieved:
        raise ValueError("No content retrieved — build the knowledge base first.")

    results: Dict[str, Dict[str, Any]] = {}

    for platform in platforms:
        if platform not in PLATFORM_PROFILES:
            results[platform] = {"error": f"Unknown platform: {platform}"}
            continue

        prompt = build_prompt(topic, platform, retrieved, tone_nudge=tone_nudge)

        try:
            parsed = call_groq(prompt)
        except (RuntimeError, ValueError) as exc:
            results[platform] = {"error": str(exc)}
            continue

        post_text = parsed.get("post", "")
        hashtags = parsed.get("hashtags", [])
        char_count = len(post_text)
        limit = PLATFORM_PROFILES[platform]["hard_char_limit"]
        exceeds_limit = limit is not None and char_count > limit

        results[platform] = {
            "post": post_text,
            "hashtags": hashtags,
            "platform": platform,
            "char_count": char_count,
            "char_limit": limit,
            "exceeds_limit": exceeds_limit,
            "sources_used": retrieved,
        }

    return results


# =============================================================================
# Theme
#
# Token system — "Slate & Sage": a cool, quiet palette built for reading and
# editing text calmly, with one deliberate signature: retrieved chunks show
# their similarity as a small filled bar instead of a bare decimal, so the
# grounding behind each post stays visible rather than reading as trivia.
#
#   bg / surface   #F4F6F5 / #FFFFFF   quiet, cool neutrals — not warm cream
#   ink / muted    #1E2B29 / #5B6B67   softened near-black, not pure black
#   accent (harbor)#2F6F62             deep muted teal — calm, not alarming
#   status colors  success/warn/error, all muted rather than saturated
#
# Type: Space Grotesk for headings (technical warmth), IBM Plex Sans for body
# (documentation-grade legibility), IBM Plex Mono for data — char counts,
# hashtags, similarity scores — so retrieved/measured things read as data.
# =============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #F4F6F5;
    --surface: #FFFFFF;
    --surface-alt: #EBEFED;
    --ink: #1E2B29;
    --ink-muted: #5B6B67;
    --border: #D9E1DE;
    --accent: #2F6F62;
    --accent-hover: #24564B;
    --accent-soft: #DCEAE5;
    --success: #3F7D5C;
    --success-soft: #DCEEE3;
    --warning: #A97417;
    --warning-soft: #F3E7D0;
    --error: #B14B3F;
    --error-soft: #F5DEDA;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stHeader"] { background-color: transparent; }

[data-testid="stSidebar"] {
    background-color: var(--surface-alt);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink); }

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
h1 { font-weight: 700 !important; }
h2, h3 { font-weight: 600 !important; }

p, span, label, div, li { font-family: 'IBM Plex Sans', sans-serif; }

.app-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.15rem;
}
.app-subtitle { color: var(--ink-muted); font-size: 1rem; }

hr { border-color: var(--border) !important; }

/* Card-style bordered containers (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.25rem 0.25rem;
    box-shadow: 0 1px 2px rgba(30, 43, 41, 0.04);
}

/* Buttons */
[data-testid="stButton"] button {
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    transition: transform 0.05s ease-in-out, box-shadow 0.15s ease-in-out;
}
[data-testid="stButton"] button:hover { transform: translateY(-1px); }
[data-testid="stButton"] button[kind="primary"] {
    background-color: var(--accent);
    border-color: var(--accent);
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: var(--accent-hover);
    border-color: var(--accent-hover);
}
[data-testid="stButton"] button[kind="secondary"] {
    background-color: var(--surface);
    border-color: var(--border);
    color: var(--ink);
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px !important;
    border-color: var(--border) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Multiselect chips */
[data-baseweb="tag"] {
    background-color: var(--accent) !important;
    border-radius: 6px !important;
}

/* Tabs */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    color: var(--ink-muted);
}
[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent);
    border-bottom-color: var(--accent) !important;
}
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background-color: var(--surface);
}

/* Native alerts, re-toned to match the palette instead of stock red/orange/blue */
[data-testid="stAlertContentError"] { color: var(--error) !important; }
[data-testid="stAlertContentWarning"] { color: var(--warning) !important; }
[data-testid="stAlertContentSuccess"] { color: var(--success) !important; }
[data-testid="stAlertContentInfo"] { color: var(--accent) !important; }

/* Custom status badges (used instead of loud alert boxes for benign status) */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    margin: 0.25rem 0 0.6rem;
}
.status-badge.neutral { background-color: var(--surface-alt); color: var(--ink-muted); border: 1px solid var(--border); }
.status-badge.ok { background-color: var(--success-soft); color: var(--success); }
.status-badge.over { background-color: var(--error-soft); color: var(--error); }

/* Relevance bars — the signature element: makes retrieval scores visible at a glance */
.source-block { margin-bottom: 0.85rem; }
.source-name {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--ink);
}
.relevance-row { display: flex; align-items: center; gap: 8px; margin: 3px 0 5px; }
.relevance-track {
    flex: 1;
    height: 6px;
    border-radius: 999px;
    background-color: var(--accent-soft);
    overflow: hidden;
}
.relevance-fill { height: 100%; border-radius: 999px; background-color: var(--accent); }
.relevance-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink-muted);
    min-width: 38px;
    text-align: right;
}
.source-preview { color: var(--ink-muted); font-size: 0.85rem; line-height: 1.4; }

/* Source legend swatches (grounding view) */
.legend-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.85rem; }
.legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.legend-label { color: var(--ink); font-weight: 500; }
.legend-score { color: var(--ink-muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }

/* Grounded / highlighted post text */
.grounding-text {
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 0.92rem;
    color: var(--ink);
    background-color: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.grounding-text mark { padding: 1px 3px; border-radius: 3px; }
.grounding-empty { color: var(--ink-muted); font-size: 0.85rem; font-style: italic; }

/* Platform preview eyebrow chip */
.preview-chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-muted);
    background-color: var(--surface-alt);
    border: 1px solid var(--border);
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Platform mockup cards */
.mockup-card {
    background-color: #FFFFFF;
    border: 1px solid #E3E6E8;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    color: #1A1A1A;
}
.mockup-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.mockup-avatar {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: #FFFFFF; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}
.mockup-name { font-weight: 600; font-size: 0.9rem; color: #1A1A1A; }
.mockup-meta { font-size: 0.78rem; color: #6B7280; }
.mockup-body { white-space: pre-wrap; font-size: 0.88rem; line-height: 1.5; color: #1A1A1A; margin-bottom: 10px; }
.mockup-image {
    width: 100%; height: 140px; border-radius: 8px; margin-bottom: 10px;
    display: flex; align-items: center; justify-content: center;
    color: rgba(255,255,255,0.85); font-size: 0.8rem; letter-spacing: 0.03em;
}
.mockup-footer { display: flex; gap: 18px; padding-top: 10px; border-top: 1px solid #EEF0F2; font-size: 0.8rem; color: #6B7280; }

/* Action row (copy / regenerate) */
.action-row-spacer { height: 4px; }
</style>
"""


# =============================================================================
# Feature: source-grounding highlighter
#
# Best-effort visual aid, not a formal citation system: uses difflib to find
# literal overlapping substrings between the generated post and each retrieved
# chunk, then highlights each match in a color tied to its source. Paraphrased
# content (the LLM's own wording) won't highlight — only near-verbatim reuse
# will. That's a reasonable trade-off for a lightweight, dependency-free check.
# =============================================================================

HIGHLIGHT_COLORS = ["#FDE68A", "#BFDBFE", "#FBCFE8", "#C7F2D8", "#E9D5FF", "#FED7AA", "#BAE6FD", "#FCA5A5"]


def build_grounding_html(post_text: str, sources_used: List[Dict[str, Any]], min_match_chars: int = 14) -> Tuple[str, List[Tuple[str, str, float]]]:
    """Highlight substrings of `post_text` that overlap with each retrieved chunk.

    Returns (highlighted_html, legend) where legend is a list of
    (color, source_name, score) for each source that had at least one match.
    """
    if not post_text.strip():
        return '<span class="grounding-empty">Nothing to show yet.</span>', []

    candidate_spans = []  # (start, end, source_idx, color)
    for idx, src in enumerate(sources_used):
        color = HIGHLIGHT_COLORS[idx % len(HIGHLIGHT_COLORS)]
        matcher = difflib.SequenceMatcher(None, post_text, src["text"], autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size >= min_match_chars:
                snippet = post_text[block.a: block.a + block.size]
                if snippet.strip():
                    candidate_spans.append((block.a, block.a + block.size, idx, color))

    if not candidate_spans:
        escaped = html_lib.escape(post_text).replace("\n", "<br>")
        return f'<div class="grounding-text">{escaped}</div>', []

    # Resolve overlaps: keep the longest matches first, skip anything that overlaps an already-kept span.
    candidate_spans.sort(key=lambda s: -(s[1] - s[0]))
    accepted: List[Tuple[int, int, int, str]] = []
    for span in candidate_spans:
        if not any(span[0] < acc[1] and acc[0] < span[1] for acc in accepted):
            accepted.append(span)
    accepted.sort(key=lambda s: s[0])

    pieces = []
    cursor = 0
    legend: List[Tuple[str, str, float]] = []
    seen_idx = set()
    for start, end, idx, color in accepted:
        pieces.append(html_lib.escape(post_text[cursor:start]))
        matched = html_lib.escape(post_text[start:end])
        source_name = html_lib.escape(str(sources_used[idx]["source"]))
        pieces.append(f'<mark style="background-color:{color};" title="From: {source_name}">{matched}</mark>')
        cursor = end
        if idx not in seen_idx:
            seen_idx.add(idx)
            legend.append((color, sources_used[idx]["source"], sources_used[idx]["score"]))

    pieces.append(html_lib.escape(post_text[cursor:]))
    highlighted = "".join(pieces).replace("\n", "<br>")
    return f'<div class="grounding-text">{highlighted}</div>', legend


# =============================================================================
# Feature: platform mockup preview
#
# Approximate, generic-identity previews (no real logos/brand marks are
# reproduced) so you can see roughly how a post will read in situ — line
# wrapping, hashtag placement, caption length — before you copy it out.
# =============================================================================


def _mockup_avatar(bg: str, initials: str = "YN") -> str:
    return f'<div class="mockup-avatar" style="background-color:{bg};">{initials}</div>'


def render_platform_mockup(platform: str, post_text: str) -> str:
    text = html_lib.escape(post_text).replace("\n", "<br>")
    chip = f'<div class="preview-chip">Preview · {html_lib.escape(platform)}</div>'

    if platform == "LinkedIn":
        card = f"""
        <div class="mockup-card">
            <div class="mockup-header">
                {_mockup_avatar('#0A66C2')}
                <div>
                    <div class="mockup-name">Your Name</div>
                    <div class="mockup-meta">Your Headline · 2h · 🌐</div>
                </div>
            </div>
            <div class="mockup-body">{text}</div>
            <div class="mockup-footer"><span>👍 Like</span><span>💬 Comment</span><span>🔁 Repost</span><span>📤 Send</span></div>
        </div>
        """
    elif platform == "X":
        card = f"""
        <div class="mockup-card">
            <div class="mockup-header">
                {_mockup_avatar('#0F1419')}
                <div>
                    <div class="mockup-name">Your Name <span class="mockup-meta">@yourhandle · 2h</span></div>
                </div>
            </div>
            <div class="mockup-body">{text}</div>
            <div class="mockup-footer"><span>💬</span><span>🔁</span><span>❤️</span><span>📊</span><span>🔖</span></div>
        </div>
        """
    elif platform == "Instagram":
        card = f"""
        <div class="mockup-card">
            <div class="mockup-header">
                {_mockup_avatar('#DD2A7B')}
                <div class="mockup-name">yourhandle</div>
            </div>
            <div class="mockup-image" style="background: linear-gradient(135deg, #F58529, #DD2A7B, #8134AF, #515BD4);">📷 your image here</div>
            <div class="mockup-footer" style="border-top:none; padding-top:0; margin-bottom:8px;"><span>❤️</span><span>💬</span><span>📤</span><span>🔖</span></div>
            <div class="mockup-body" style="margin-bottom:0;"><span class="mockup-name">yourhandle</span> {text}</div>
        </div>
        """
    elif platform == "Facebook":
        card = f"""
        <div class="mockup-card">
            <div class="mockup-header">
                {_mockup_avatar('#1877F2')}
                <div>
                    <div class="mockup-name">Your Name</div>
                    <div class="mockup-meta">2h · 🌐</div>
                </div>
            </div>
            <div class="mockup-body">{text}</div>
            <div class="mockup-footer"><span>👍 Like</span><span>💬 Comment</span><span>↗️ Share</span></div>
        </div>
        """
    else:  # Threads
        card = f"""
        <div class="mockup-card">
            <div class="mockup-header">
                {_mockup_avatar('#101010')}
                <div class="mockup-name">yourhandle <span class="mockup-meta">2h</span></div>
            </div>
            <div class="mockup-body">{text}</div>
            <div class="mockup-footer"><span>♡</span><span>💬</span><span>🔁</span><span>↗</span></div>
        </div>
        """
    return chip + card


# =============================================================================
# Feature: real copy-to-clipboard button (st.markdown can't execute <script>
# tags, so this uses components.html, which renders in its own iframe and
# does run JS)
# =============================================================================


def copy_button_component(text: str, element_id: str) -> None:
    # Safely embed as a JS string literal, and neutralize "</" so a post that happens to
    # contain a literal "</script>" can't prematurely close this component's <script> tag.
    safe_text = json.dumps(text).replace("</", "<\\/")
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "-", element_id)
    snippet = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <button id="copy-{safe_id}" style="
          background-color:#2F6F62; color:#FFFFFF; border:none; border-radius:8px;
          padding:8px 14px; font-size:0.85rem; font-weight:500; cursor:pointer;
          display:inline-flex; align-items:center; gap:6px; transition:background-color 0.15s;
      " onmouseover="this.style.backgroundColor='#24564B'" onmouseout="this.style.backgroundColor='#2F6F62'">
        📋 Copy post
      </button>
    </div>
    <script>
      const btn = document.getElementById("copy-{safe_id}");
      btn.addEventListener("click", function() {{
        navigator.clipboard.writeText({safe_text}).then(function() {{
          btn.innerHTML = "✓ Copied";
          setTimeout(function() {{ btn.innerHTML = "📋 Copy post"; }}, 1500);
        }}).catch(function() {{
          btn.innerHTML = "Couldn't copy — select manually";
          setTimeout(function() {{ btn.innerHTML = "📋 Copy post"; }}, 2000);
        }});
      }});
    </script>
    """
    components.html(snippet, height=46)


def relevance_bar_html(score: float) -> str:
    """Render a chunk's similarity score as a small filled bar + mono value."""
    pct = max(0, min(100, round(score * 100)))
    return (
        f'<div class="relevance-row">'
        f'<div class="relevance-track"><div class="relevance-fill" style="width:{pct}%;"></div></div>'
        f'<span class="relevance-score">{score:.2f}</span>'
        f'</div>'
    )


def char_status_badge_html(char_count: int, limit: Optional[int]) -> str:
    """Render the character-count status as a quiet inline badge rather than a loud alert box."""
    if limit is None:
        return f'<span class="status-badge neutral">{char_count} chars · no hard cap for this platform</span>'
    if char_count > limit:
        return f'<span class="status-badge over">⚠ {char_count} / {limit} · over the limit</span>'
    return f'<span class="status-badge ok">✓ {char_count} / {limit} · within limit</span>'


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="MultiPlatform RAG Post Generator", page_icon="🧭", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="app-eyebrow">Retrieval-grounded content studio</div>', unsafe_allow_html=True)
st.title("🧭 MultiPlatform RAG Post Generator")
st.markdown(
    '<div class="app-subtitle">Ground your social posts in your own files or URLs. Retrieval finds the '
    "relevant chunks; Groq only ever sees those chunks, your topic, and the selected platform's profile.</div>",
    unsafe_allow_html=True,
)
st.write("")

# --- Resolve the Groq API key (st.secrets, never hardcoded) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error(
        "⚠️ **GROQ_API_KEY not found.** Add it under your app's **Settings → Secrets** in Streamlit "
        "Community Cloud (or in a local `.streamlit/secrets.toml` when running locally):\n\n"
        '```toml\nGROQ_API_KEY = "your-real-key-here"\n```'
    )


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


groq_client = get_groq_client(GROQ_API_KEY) if GROQ_API_KEY else None

# --- Session state defaults ---
_defaults = {
    "kb_built": False,
    "vector_store": None,
    "kb_summary": None,
    "generation_results": {},
    "last_topic": "the key takeaways",
    "last_tone_nudge": "",
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# --- Sidebar controls ---
with st.sidebar:
    st.markdown('<div class="app-eyebrow">Controls</div>', unsafe_allow_html=True)
    st.header("Knowledge base settings")

    st.caption("Retrieval tuning")
    chunk_size = st.slider("Chunk size (characters)", min_value=200, max_value=2000, value=800, step=50)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=500, value=150, step=10)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 50)
        st.caption(f"Overlap must be smaller than chunk size — adjusted to {chunk_overlap}.")
    top_k = st.slider("Chunks retrieved per post (top-k)", min_value=1, max_value=10, value=3)

    st.divider()
    if st.button("🗑️ Clear knowledge base / Reset", use_container_width=True):
        st.session_state.kb_built = False
        st.session_state.vector_store = None
        st.session_state.kb_summary = None
        st.session_state.generation_results = {}
        for key in list(st.session_state.keys()):
            if key.startswith("post_text_"):
                del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption(
        "Streamlit Community Cloud's free tier has an ephemeral filesystem, so the knowledge base "
        "is rebuilt fresh each session and is not saved between visits."
    )

st.write("")

# --- Step 1: source input ---
st.subheader("1. Add your source content")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload files (.txt / .pdf)", type=["txt", "pdf"], accept_multiple_files=True
        )
    with col2:
        url_block = st.text_area(
            "Or paste URLs (one per line)",
            height=150,
            placeholder="https://example.com/article-one\nhttps://example.com/article-two",
        )

    has_input = bool(uploaded_files) or bool(url_block and url_block.strip())
    build_clicked = st.button("📚 Build knowledge base", type="primary", disabled=not has_input)
    if not has_input:
        st.caption("Upload at least one file or paste at least one URL to enable this button.")

    if build_clicked:
        with st.spinner("Ingesting content..."):
            documents, ingest_errors = ingest_sources(uploaded_files, url_block)

        for err in ingest_errors:
            st.warning(err)

        if not documents:
            st.error("No usable content was ingested. Fix the issues above and try again.")
        else:
            try:
                with st.spinner("Loading embedding model, chunking, and indexing..."):
                    model = load_embedding_model()
                    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
                    if not chunks:
                        raise ValueError("Ingested content produced zero chunks — check your chunk size settings.")
                    vector_store = VectorStore(model)
                    vector_store.build(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.kb_built = True
                st.session_state.kb_summary = {
                    "num_sources": len(documents),
                    "num_chunks": len(chunks),
                    "sources": list(documents.keys()),
                }
                st.session_state.generation_results = {}
                st.success(f"✅ Knowledge base built: {len(chunks)} chunks from {len(documents)} source(s).")
            except Exception as exc:
                st.error(f"Failed to build the knowledge base: {exc}")

    if st.session_state.kb_built and st.session_state.kb_summary:
        summary = st.session_state.kb_summary
        with st.expander(f"📖 Current knowledge base — {summary['num_chunks']} chunks from {summary['num_sources']} source(s)"):
            for s in summary["sources"]:
                st.write(f"- {s}")

st.write("")

# --- Step 2: topic + platforms ---
st.subheader("2. Choose your topic and platforms")
with st.container(border=True):
    topic_input = st.text_input("What should the post focus on?", placeholder="the key takeaways")
    topic = topic_input.strip() if topic_input and topic_input.strip() else "the key takeaways"

    tone_nudge_input = st.text_input(
        "Add a tone nudge (optional)",
        placeholder="e.g. more playful, more formal, add a personal anecdote",
        help="Layered on top of each platform's built-in tone/length/hashtag rules — it won't override them.",
    )

    platforms_selected = st.multiselect("Platforms", options=list(PLATFORM_PROFILES.keys()))

    generate_disabled = not (st.session_state.kb_built and platforms_selected and GROQ_API_KEY)
    generate_clicked = st.button("✨ Generate Posts", type="primary", disabled=generate_disabled)

    if not st.session_state.kb_built:
        st.caption("Build a knowledge base above before generating posts.")
    elif not platforms_selected:
        st.caption("Select at least one platform to generate posts for.")
    elif not GROQ_API_KEY:
        st.caption("Add GROQ_API_KEY in Streamlit secrets to enable generation.")

    if generate_clicked:
        if groq_client is None:
            st.error("Groq client is not configured — GROQ_API_KEY is missing.")
        else:
            with st.spinner(f"Retrieving context and generating {len(platforms_selected)} post(s)..."):
                try:
                    results = generate_posts(
                        st.session_state.vector_store, topic, platforms_selected, k=top_k, tone_nudge=tone_nudge_input
                    )
                    st.session_state.generation_results = results
                    st.session_state.last_topic = topic
                    st.session_state.last_tone_nudge = tone_nudge_input
                    for key in list(st.session_state.keys()):
                        if key.startswith("post_text_"):
                            del st.session_state[key]
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

st.write("")

# --- Step 3: results ---
st.subheader("3. Your posts")
with st.container(border=True):
    results = st.session_state.generation_results

    if results:
        # Export current (possibly edited) posts for every platform that generated successfully.
        exportable = {
            platform: st.session_state.get(f"post_text_{platform}", data.get("post", ""))
            for platform, data in results.items()
            if "error" not in data
        }
        if exportable:
            txt_parts = []
            for platform, text in exportable.items():
                txt_parts.append(f"=== {platform} ===\n{text}\n")
            txt_blob = "\n".join(txt_parts)

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["platform", "post", "hashtags", "char_count", "char_limit", "exceeds_limit"])
            for platform, text in exportable.items():
                data = results[platform]
                writer.writerow([
                    platform,
                    text,
                    ", ".join(data.get("hashtags", [])),
                    len(text),
                    data.get("char_limit", ""),
                    data.get("exceeds_limit", ""),
                ])

            exp_col1, exp_col2, _ = st.columns([1, 1, 3])
            with exp_col1:
                st.download_button(
                    "📄 Export .txt", data=txt_blob, file_name="posts.txt", mime="text/plain", use_container_width=True
                )
            with exp_col2:
                st.download_button(
                    "📊 Export .csv", data=csv_buffer.getvalue(), file_name="posts.csv", mime="text/csv", use_container_width=True
                )
            st.write("")

        tabs = st.tabs(list(results.keys()))
        for tab, platform in zip(tabs, results.keys()):
            data = results[platform]
            with tab:
                if "error" in data:
                    st.error(f"{platform} failed: {data['error']}")
                    regen_key = f"regen_err_{platform}"
                    if st.button("🔁 Try again", key=regen_key):
                        if groq_client is None:
                            st.error("Groq client is not configured — GROQ_API_KEY is missing.")
                        else:
                            retry_topic = st.session_state.get("last_topic", topic)
                            retry_nudge = st.session_state.get("last_tone_nudge", tone_nudge_input)
                            with st.spinner(f"Regenerating {platform}..."):
                                try:
                                    single = generate_posts(
                                        st.session_state.vector_store, retry_topic, [platform], k=top_k, tone_nudge=retry_nudge
                                    )
                                    st.session_state.generation_results[platform] = single[platform]
                                    st.session_state.pop(f"post_text_{platform}", None)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Regeneration failed: {exc}")
                    continue

                text_key = f"post_text_{platform}"
                if text_key not in st.session_state:
                    st.session_state[text_key] = data["post"]

                st.text_area("Post text (editable)", key=text_key, height=200)
                current_text = st.session_state[text_key]
                char_count = len(current_text)
                limit = data["char_limit"]

                action_col1, action_col2, action_col3 = st.columns([1.1, 1.3, 2.6])
                with action_col1:
                    copy_button_component(current_text, element_id=f"copy_{platform}")
                with action_col2:
                    if st.button("🔁 Regenerate", key=f"regen_{platform}", use_container_width=True):
                        if groq_client is None:
                            st.error("Groq client is not configured — GROQ_API_KEY is missing.")
                        else:
                            retry_topic = st.session_state.get("last_topic", topic)
                            retry_nudge = st.session_state.get("last_tone_nudge", tone_nudge_input)
                            with st.spinner(f"Regenerating {platform}..."):
                                try:
                                    single = generate_posts(
                                        st.session_state.vector_store, retry_topic, [platform], k=top_k, tone_nudge=retry_nudge
                                    )
                                    st.session_state.generation_results[platform] = single[platform]
                                    st.session_state.pop(text_key, None)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Regeneration failed: {exc}")

                st.markdown(char_status_badge_html(char_count, limit), unsafe_allow_html=True)
                st.write("**Hashtags:**", ", ".join(data["hashtags"]) if data["hashtags"] else "_none suggested_")

                st.write("")
                st.markdown("**Preview**")
                st.markdown(render_platform_mockup(platform, current_text), unsafe_allow_html=True)

                with st.expander("🔍 Grounding — see which sources this post draws from"):
                    highlighted_html, legend = build_grounding_html(current_text, data["sources_used"])
                    st.markdown(highlighted_html, unsafe_allow_html=True)

                    if legend:
                        st.caption("Highlighted phrases trace back to:")
                        for color, source_name, score in legend:
                            st.markdown(
                                f'<div class="legend-row">'
                                f'<span class="legend-swatch" style="background-color:{color};"></span>'
                                f'<span class="legend-label">{html_lib.escape(str(source_name))}</span>'
                                f'<span class="legend-score">similarity {score:.2f}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No near-verbatim phrases matched — the model paraphrased the source material.")

                    st.divider()
                    st.caption("All retrieved sources for this post:")
                    for src in data["sources_used"]:
                        preview = src["text"][:300] + ("..." if len(src["text"]) > 300 else "")
                        st.markdown(
                            f'<div class="source-block">'
                            f'<div class="source-name">{html_lib.escape(str(src["source"]))}</div>'
                            f'{relevance_bar_html(src["score"])}'
                            f'<div class="source-preview">{html_lib.escape(preview)}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.caption("Generated posts will appear here after you build a knowledge base and click Generate Posts.")
