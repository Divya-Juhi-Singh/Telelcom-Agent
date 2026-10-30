"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three possible sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)
- Guide documents (technical guides and troubleshooting instructions)

CITATION RULES:
1. Answer using only the retrieved context.
2. Every factual statement must include a citation.
3. FAQ information must cite the FAQ question or title.
4. Support ticket information must cite its exact ticket ID.
5. Guide information must cite the guide name and page number.
6. Place citations immediately after the statement they support.
7. If multiple sources support a statement, cite all relevant sources.
8. Never invent ticket IDs, FAQ titles, guide names, or page numbers.
9. If a source does not contain a usable identifier, do not invent one.
10. If the context is insufficient, clearly say so and suggest calling 611
    or using the MyTelecom app.
    
Example answer:
To reset your network settings, open Settings → General → Transfer or Reset
→ Reset → Reset Network Settings [FAQ: Network Troubleshooting].

If a similar issue was resolved previously, you can also try restarting
your device after resetting the network settings [Ticket: TICKET-1024].

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []

    for doc in docs:
        metadata = doc.metadata
        source_type = metadata.get("source_type", "UNKNOWN")

        if source_type == "FAQ":
            citation = (
                f"[FAQ: {metadata.get('question', metadata.get('title', 'FAQ Entry'))}]"
            )

        elif source_type == "TICKET":
            ticket_id = metadata.get("ticket_id", metadata.get("id", "Unknown ID"))
            citation = f"[Ticket ID: {ticket_id}]"

        elif source_type == "GUIDE":
            guide_name = metadata.get(
                "source_name",
                metadata.get("source", "Technical Guide")
            )
            page = metadata.get("page")

            if page is not None:
                citation = f"[Guide: {guide_name}, Page: {page + 1}]"
            else:
                citation = f"[Guide: {guide_name}]"

        else:
            citation = "[Source: Unknown]"

        sections.append(
            f"{citation}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)
def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
