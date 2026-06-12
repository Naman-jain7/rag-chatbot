import subprocess

frontend = subprocess.Popen(["uv", "run", "streamlit", "run", "streamlit_app.py"])

backend = subprocess.Popen(
    ["uv", "run", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
)

try:
    frontend.wait()
    backend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()
