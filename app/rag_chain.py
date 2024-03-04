from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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

final_chain = ANSWER_PROMPT | llm | StrOutputParser()

FINAL_CHAIN_INVOKE = final_chain.invoke({"context": "", "question": "what are examples of distributed system"})

print(FINAL_CHAIN_INVOKE)
