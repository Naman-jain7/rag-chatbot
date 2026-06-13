from langsmith import Client
from app.core.config import settings




client = Client()

run = client.read_run("id")

print(run.total_tokens)
