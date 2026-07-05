import subprocess

frontend = subprocess.Popen(["uv", "run", "streamlit", "run", "streamlit_app.py"])

backend = subprocess.Popen(
    ["uv", "run", "python", "run_backend.py"]
)

try:
    frontend.wait()
    backend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()
