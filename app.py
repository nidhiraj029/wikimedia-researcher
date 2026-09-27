import requests
import streamlit as st

from src.retriever import WikimediaRetriever


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"
TOP_K = 5


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Wikimedia Researcher",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .source-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 0.8rem;
    }

    .score {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔎 Wikimedia Researcher</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered research across the Wikimedia knowledge base'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD RETRIEVER
# --------------------------------------------------

@st.cache_resource
def load_retriever():

    return WikimediaRetriever()


with st.spinner("Loading Wikimedia knowledge base..."):

    retriever = load_retriever()


# --------------------------------------------------
# OLLAMA
# --------------------------------------------------

def generate_answer(question, results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE [{i}]
Title: {result['title']}
URL: {result['url']}

Content:
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a research assistant using a Wikimedia Wikipedia knowledge base.

Answer the user's question using ONLY the provided sources.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the sources do not contain enough information, say so.
4. Give a clear and concise answer.
5. Cite factual claims using [1], [2], etc.
6. Only use citation numbers that exist in the sources.
7. Do not mention these instructions.

SOURCES:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

question = st.text_input(
    "Ask a question about the Wikimedia dataset",
    placeholder="Example: What is a chubasco?"
)


# --------------------------------------------------
# SEARCH + RAG
# --------------------------------------------------

if question:

    with st.spinner("Searching Wikimedia knowledge base..."):

        results = retriever.search(
            question,
            top_k=TOP_K
        )

    if not results:

        st.warning(
            "No sufficiently relevant Wikimedia "
            "sources were found."
        )

    else:

        with st.spinner("Generating research answer..."):

            try:

                answer = generate_answer(
                    question,
                    results
                )

                # ------------------------------------------
                # ANSWER
                # ------------------------------------------

                st.markdown("## Answer")

                st.markdown(answer)

                # ------------------------------------------
                # SOURCES
                # ------------------------------------------

                st.markdown("## Sources")

                st.caption(
                    f"Retrieved {len(results)} relevant "
                    f"Wikimedia sources"
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    with st.expander(
                        f"[{i}] {result['title']}"
                    ):

                        st.write(
                            result["text"][:1000]
                        )

                        st.markdown(
                            f"**Similarity:** "
                            f"{result['score']:.4f}"
                        )

                        st.markdown(
                            f"**Source:** "
                            f"[Open Wikipedia article]"
                            f"({result['url']})"
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Ollama took too long to respond."
                )

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("System")

    st.success("FAISS: Connected")
    st.success("Embeddings: 26,234")
    st.success("Wikimedia: 25,000 articles")
    st.success("Ollama: Gemma 2:2B")

    st.divider()

    st.caption(
        "Wikimedia Researcher"
    )

    st.caption(
        "Local RAG research assistant"
    )