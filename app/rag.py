"""Retrieval-Augmented Generation chain with numbered source citations."""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

SYSTEM_PROMPT = """You are a friendly teaching assistant for an Artificial Intelligence course.
Answer the trainee's question using ONLY the numbered course context below.

Rules:
- Reply in the same language as the question (Arabic or English).
- Explain simply, as a good instructor would, and give a short example when helpful.
- Cite the context you used with its number in square brackets, e.g. [1] or [2].
- If the answer is not in the context, say that the course material does not cover it. Do not invent facts.

Course context:
{context}"""

NO_CONTEXT_ANSWER = (
    "I couldn't find this in the course material. / "
    "لم أجد إجابة لهذا السؤال في مادة المقرر."
)


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        where = f"{source}, page {page + 1}" if isinstance(page, int) else source
        parts.append(f"[{i}] ({where})\n{doc.page_content}")
    return "\n\n".join(parts)


def to_sources(docs: list[Document], snippet_len: int = 220) -> list[dict]:
    sources = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        sources.append(
            {
                "id": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": page + 1 if isinstance(page, int) else None,
                "snippet": doc.page_content[:snippet_len].strip(),
            }
        )
    return sources


class RAGAssistant:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, top_k: int = 4):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
        self.chain = prompt | llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        docs = self.retriever.invoke(question)
        if not docs:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}
        answer = self.chain.invoke({"context": format_docs(docs), "question": question})
        return {"answer": answer.strip(), "sources": to_sources(docs)}
