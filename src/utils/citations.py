import re
from typing import Any, Dict

def format_citations(state: Dict[str, Any]) -> str:
    """
    Extracts inline citations like [filename.ext - Page X] from the LLM response
    and formats them into a neat reference list at the bottom.
    """
    response = state.get("response", "")
    if not response:
        return ""
        
    # Regex to find [Filename - Page X]
    # Example match: ('budget.pdf', '3')
    pattern = r"\[(.*?) - Page (.*?)\]"
    matches = re.findall(pattern, response)
    
    if not matches:
        return ""
        
    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for filename, page in matches:
        citation = f"- {filename.strip()} (Page {page.strip()})"
        if citation not in seen:
            seen.add(citation)
            unique_citations.append(citation)
            
    if not unique_citations:
        return ""
        
    formatted_refs = "\n\n---\n**References:**\n" + "\n".join(unique_citations)
    return formatted_refs
