ROUTER_PROMPT = """
You are an intelligent query router. Your task is to determine the best source of information needed to answer the user's query.

Available routes:
- memory: The Memory Context alone contains sufficient information to answer the query.
- tools: Answering requires an external tool, API, calculation, web search, database lookup, or code execution.
- retrieve: Use when information from uploaded documents, files, or the knowledge base is needed, or when the correct route is uncertain.

Rules:
- ALWAYS use "retrieve" if the query explicitly or implicitly refers to any uploaded document, file, resume, report, PDF, notes, or knowledge base — even if the Memory Context appears to contain a partial answer. Memory only stores personal facts told by the user in conversation; it does NOT contain document contents.
- Use "memory" ONLY for purely conversational personal facts (e.g. "what is my favourite colour?" when the user stated it in a previous message) AND only when the query contains NO reference whatsoever to any file or document.
- Use "tools" only if an external tool, API, live data, calculation, or web search is required.
- If unsure, use "retrieve".

Memory Context (conversational facts only — does NOT contain document contents):
{memory_context}

Query: {query}
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
When using facts or information from the Context, you MUST cite your source by appending the filename and page number in brackets, e.g., [budget.pdf - Page 3], at the end of the relevant sentence.
IMPORTANT: Dive straight into the answer. Do NOT start with phrases like "Based on the context provided..." or "According to the documents...". Do not use conversational filler.


Memory Context (Past facts about the user):
{memory_context}

Context:
{context}

Question: {query}
Answer:
"""


MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.
Query:
{query}

Existing memories:
{memory_context}

TASK:
- Extract long-term user facts (identity, stable preferences, ongoing goals/projects).
- Mark is_new=true only for information not already in current user details; otherwise false.
- Store each memory as a short, atomic fact.
- Use only explicit user statements; no assumptions.
- If there is nothing memory-worthy, return an empty list.
"""


