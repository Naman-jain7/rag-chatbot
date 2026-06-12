from langsmith import Client
from app.core.config import settings

def get_token_usage(session_id: str)->dict:
    client = Client()
    runs = client.list_runs(
        project_name=settings.app_config.LANGCHAIN_PROJECT,
        limit=1,
        filter=f'and(eq(metadata_key, "session_id"), eq(metadata_value, "{session_id}"))',
    )

    latest_run = next(runs, None)

    if latest_run is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    run = client.read_run(latest_run.id)

    return {
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "total_tokens": run.total_tokens
    }