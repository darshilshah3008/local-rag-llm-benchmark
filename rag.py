import argparse
from typing import List, Tuple, Dict

from utils import call_lmstudio
from embedding import ingest_and_build_index, load_index
from search import Retriever


BASE_SYSTEM_PROMPT = """
You are an expert research assistant working over a collection of PDF documents.
You MUST answer ONLY using the provided context excerpts.
If the context is insufficient, say so clearly.

When the user asks "what is this PDF about" or similar:
- Give a high-level overview in 3–6 bullet points.
- Mention the main goal of the project/assignment.
- Mention the target audience and context (e.g., Indian professionals, non-tech learners).
- Mention the key sections or deliverables.
- Mention any important constraints (like submission format, page limits, file naming).

Always be structured and concise.
When relevant, mention the file name and page number.
If tables are involved, explain them logically in text form.
"""



# -------------------------------------------------------------
# Build context block that matches NotebookLM-style chunks
# -------------------------------------------------------------
def build_context_block(chunks_with_scores: List[Tuple[Dict, float]]) -> str:
    parts = []
    for i, (meta, score) in enumerate(chunks_with_scores, start=1):
        header = (
            f"### Excerpt {i} | "
            f"file={meta['pdf_name']} | "
            f"page={meta['page_num']} | "
            f"type={meta['unit_type']} | "
            f"chunk={meta['chunk_index']} | "
            f"similarity={score:.3f}"
        )

        body = meta["chunk_text"]

        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)



# -------------------------------------------------------------
# Main question-answer helper
# -------------------------------------------------------------
def answer_question(retriever: Retriever, question: str, k: int = 8) -> str:
    retrieved = retriever.retrieve(question, k=k)

    if not retrieved:
        return "I couldn't find any relevant context in the documents."

    context_block = build_context_block(retrieved)

    user_prompt = f"""
You are given the following excerpts from one or more PDF documents:

{context_block}

Now answer this question, using ONLY the information in the excerpts above:

{question}

If the excerpts do not contain enough information, explicitly say that you do not know.
Whenever useful, reference file names and page numbers.
"""

    answer = call_lmstudio(BASE_SYSTEM_PROMPT, user_prompt)
    return answer


# -------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NotebookLM-style local RAG using LM Studio + Mistral"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest all PDFs in data/pdfs and rebuild the FAISS index",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_and_build_index()
        print("\n[OK] Ingestion complete.\n")

    # Load existing index
    doc_index = load_index()
    retriever = Retriever(doc_index)

    print("RAG is ready. Ask questions about your PDFs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Q> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        answer = answer_question(retriever, q)
        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    main()
