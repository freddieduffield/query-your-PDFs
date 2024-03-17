import os
from operator import itemgetter
from typing import TypedDict

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import PG_COLLECTION_NAME

vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string="postgresql+psycopg://postgres@localhost:5432/pdf_rag_vectors",
    embedding_function=OpenAIEmbeddings()
)

template = """
Answer given following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    temperature=0,
    model='gpt-4-1106-preview',
    streaming=True,
)


class RagInput(TypedDict):
    question: str


final_chain = (
    RunnableParallel(
        context=(itemgetter("question") | vector_store.as_retriever()),
        question=itemgetter("question")
    ) |
    RunnableParallel(
        answer=(ANSWER_PROMPT | llm),
        docs=itemgetter("context")
    )
).with_types(input_type=RagInput)
