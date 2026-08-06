"""
MultiPlatform RAG Post Generator — Streamlit app (Phase 2)

Wraps the exact ingestion -> chunking -> local-embedding -> FAISS -> retrieval
-> platform-profiled Groq generation pipeline validated in Phase 1's Colab
notebook (multiplatform_rag_post_generator_test.ipynb) in an upload/URL +
platform-picker interface. No pipeline logic is rewritten here — only the
key-resolution and file-reading glue changes to fit Streamlit.
"""

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
import streamlit as st
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
# Platform-profiled prompt builder (identical to Phase 1)
# =============================================================================


def build_prompt(topic: str, platform: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    if platform not in PLATFORM_PROFILES:
        raise ValueError(f"Unknown platform: {platform}")

    profile = PLATFORM_PROFILES[platform]
    context_block = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks)

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


def generate_posts(vector_store: VectorStore, topic: str, platforms: List[str], k: int = 3) -> Dict[str, Dict[str, Any]]:
    """Retrieve once for `topic`, then generate one platform-tailored post per entry in `platforms`."""
    retrieved = vector_store.search(topic, k=k)
    if not retrieved:
        raise ValueError("No content retrieved — build the knowledge base first.")

    results: Dict[str, Dict[str, Any]] = {}

    for platform in platforms:
        if platform not in PLATFORM_PROFILES:
            results[platform] = {"error": f"Unknown platform: {platform}"}
            continue

        prompt = build_prompt(topic, platform, retrieved)

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
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="MultiPlatform RAG Post Generator", page_icon="📝", layout="wide")

st.title("📝 MultiPlatform RAG Post Generator")
st.caption(
    "Ground your social posts in your own files or URLs. Retrieval finds the relevant chunks; "
    "Groq only ever sees those chunks, your topic, and the selected platform's profile."
)

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
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Knowledge base settings")
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

# --- Step 1: source input ---
st.subheader("1. Add your source content")
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

st.divider()

# --- Step 2: topic + platforms ---
st.subheader("2. Choose your topic and platforms")

topic_input = st.text_input("What should the post focus on?", placeholder="the key takeaways")
topic = topic_input.strip() if topic_input and topic_input.strip() else "the key takeaways"

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
                results = generate_posts(st.session_state.vector_store, topic, platforms_selected, k=top_k)
                st.session_state.generation_results = results
                for key in list(st.session_state.keys()):
                    if key.startswith("post_text_"):
                        del st.session_state[key]
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

st.divider()

# --- Step 3: results ---
st.subheader("3. Your posts")

results = st.session_state.generation_results
if results:
    tabs = st.tabs(list(results.keys()))
    for tab, platform in zip(tabs, results.keys()):
        data = results[platform]
        with tab:
            if "error" in data:
                st.error(f"{platform} failed: {data['error']}")
                continue

            text_key = f"post_text_{platform}"
            if text_key not in st.session_state:
                st.session_state[text_key] = data["post"]

            st.text_area("Post text (editable — select all to copy)", key=text_key, height=220)
            current_text = st.session_state[text_key]
            char_count = len(current_text)
            limit = data["char_limit"]

            if limit is None:
                st.info(f"Character count: **{char_count}** — {platform} has no hard cap; follow the profile's length guidance.")
            elif char_count > limit:
                st.error(f"⚠️ Character count: **{char_count} / {limit}** — exceeds {platform}'s limit.")
            else:
                st.success(f"✅ Character count: **{char_count} / {limit}** — within {platform}'s limit.")

            st.write("**Hashtags:**", ", ".join(data["hashtags"]) if data["hashtags"] else "_none suggested_")

            with st.expander("Sources used"):
                for src in data["sources_used"]:
                    st.markdown(f"**{src['source']}** — similarity {src['score']:.3f}")
                    preview = src["text"][:300] + ("..." if len(src["text"]) > 300 else "")
                    st.caption(preview)
else:
    st.caption("Generated posts will appear here after you build a knowledge base and click Generate Posts.")
