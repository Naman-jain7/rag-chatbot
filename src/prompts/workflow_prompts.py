EVALUATE_EVIDENCE_PROMPT = """You are a strict evaluator. Read the context and the user query.

Query: {query}

Context:
{context}

Determine if the Context contains sufficient information to comprehensively answer the Query.
Respond strictly in accordance to the output schema provided.
"""

REWRITE_QUERY_PROMPT = """The user query '{query}' did not return sufficient evidence in our vector database.
Please rewrite the query to be broader, or use different synonyms to improve retrieval chances.
Return ONLY the rewritten query text and nothing else.
"""

GENERATE_ANSWER_PROMPT = """Use the following Context and Memory Context to answer the Question.
If the context is empty or unhelpful, state that you do not have enough information.

Memory Context (Past facts about the user):
{memory_context}

Context:
{context}

Question: {query}
Answer:"""
