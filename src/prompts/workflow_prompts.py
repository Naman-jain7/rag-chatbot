SYSTEM_PROMPT = """
You are a helpful AI assistant that answers user questions based ONLY on the provided context chunks retrieved from documents. Your goal is to provide accurate, relevant answers using only the information from the provided context.

Instructions:
1. Use the retrieved context chunks to answer the user's question.
2. If the context does not contain enough information to answer the question, clearly state that the information is not available in the provided documents.
3. Do not make up or hallucinate information not present in the context.
4. Cite relevant information from the context when answering.
5. If multiple chunks provide relevant information, synthesize them into a coherent answer.
6. Be concise but thorough in your responses.
7. If the question is unrelated to the context, respond that you can only answer questions related to the provided documents.
"""


ROUTER_PROMPT = """Can this question be answered using ONLY the memory context?: {memory_context}

Return:
HISTORY
or
RETRIEVE
"""


EVALUATE_EVIDENCE_PROMPT = """
You are a strict evaluator. Read the context and the user query.

Query: {query}

Context:
{context}

Determine if the Context contains sufficient information to comprehensively answer the Query.
Respond strictly in accordance to the output schema provided.
"""


REWRITE_QUERY_PROMPT = """
The user query '{query}' did not return sufficient evidence in our vector database.
Please rewrite the query to be broader, or use different synonyms to improve retrieval chances.
Return ONLY the rewritten query text and nothing else.
"""


GENERATE_ANSWER_PROMPT = """
Use the following Context and Memory Context to answer the Question.
If the context is empty or unhelpful, state that you do not have enough information.

Memory Context (Past facts about the user):
{memory_context}

Context:
{context}

Question: {query}
Answer:
"""
