import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

SYSTEM_MESSAGE = """You are a helpful AI assistant that answers user questions based ONLY on the provided context chunks retrieved from documents. Your goal is to provide accurate, relevant answers using only the information from the provided context.

Instructions:
1. Use the retrieved context chunks to answer the user's question.
2. If the context does not contain enough information to answer the question, clearly state that the information is not available in the provided documents.
3. Do not make up or hallucinate information not present in the context.
4. Cite relevant information from the context when answering.
5. If multiple chunks provide relevant information, synthesize them into a coherent answer.
6. Be concise but thorough in your responses.
7. If the question is unrelated to the context, respond that you can only answer questions related to the provided documents.
"""

# Human message template
HUMAN_TEMPLATE = """Question: {question}

Context:
{context}

Answer the question based ONLY on the context provided above.
"""

# Create ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ]
)

with open("system_prompts.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "system_message": SYSTEM_MESSAGE,
            "human_template": HUMAN_TEMPLATE,
            "input_variables": ["question", "context"],
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
