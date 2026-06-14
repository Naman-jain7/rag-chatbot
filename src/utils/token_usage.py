from langsmith import Client
from app.core.config import settings

def get_token_usage(user_id: int | str, session_id: str)->dict:
    client = Client()
    
    latest_run = next(
        client.list_runs(
            project_name=settings.app.LANGCHAIN_PROJECT,
            is_root=True,
            limit=1,
            filter=f'and(eq(metadata_key, "thread_id"), eq(metadata_value, "{session_id}"))',
        ),
        None,
    )

    if latest_run is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0


    for run in client.list_runs(trace_id=latest_run.trace_id):
        if run.run_type != "llm":
            continue
        
        input_tokens += run.input_tokens or 0
        output_tokens += run.output_tokens or 0
        total_tokens += run.total_tokens or 0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }