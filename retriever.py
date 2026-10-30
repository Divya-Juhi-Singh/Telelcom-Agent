"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma returns distances: lower = more similar.
MAX_DISTANCE = 0.8


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
    max_distance: float = MAX_DISTANCE,
) -> RunnableLambda:

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    def retrieve(query: str) -> list[Document]:

        collections = [
            (faq_store, k_faq, "FAQ"),
            (tickets_store, k_tickets, "TICKET"),
            (guides_store, k_guides, "GUIDE"),
        ]

        filtered_docs = []

        for store, k, source_type in collections:

            results = store.similarity_search_with_score(
                query,
                k=k,
            )

            for doc, distance in results:

                print(
                    f"{source_type} | Distance: {distance:.4f}"
                )

                if distance <= max_distance:

                    doc.metadata["source_type"] = source_type
                    doc.metadata["similarity_distance"] = float(distance)

                    filtered_docs.append(doc)

        print(f"\nRetrieved documents after filtering: {len(filtered_docs)}")

        return filtered_docs

    return RunnableLambda(retrieve)