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
Use the following Context to answer the Question.
If the context is empty or unhelpful, state that you do not have enough information.
When using facts or information from the Context, you MUST cite your source(s) by appending the document name and page number(s) in brackets at the end of the relevant sentence.
For example: [budget.pdf - Page 3] or [budget.pdf - Pages 3, 5].
If the document does not have a page number (like an audio file), cite just the filename: [interview.m4a].

Context:
{context}

Question: {query}
Answer:
"""

CHECK_SHORT_TERM_PROMPT = """
You are a context evaluator.
Read the conversation history and the latest user query.
Determine if the conversation history alone provides enough explicit information to directly answer the latest query.

For example:
- If history says "My name is John" and query is "What is my name?", this is SUFFICIENT (True).
- If history says "Hello" and query is "What is the capital of France?", this is INSUFFICIENT (False).

Conversation History:
{chat_history}

Latest Query: {query}

Is the history sufficient to answer the query? Respond strictly using the output schema.
"""