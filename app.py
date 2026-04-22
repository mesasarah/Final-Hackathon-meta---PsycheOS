"""
PsycheOS — HuggingFace Space Entrypoint
This file is the root app.py for HF Spaces deployment.
"""

import subprocess
import sys
import os

# Install dependencies if needed
def install_deps():
    deps = [
        "streamlit",
        "langgraph",
        "langchain",
        "faiss-cpu",
        "sentence-transformers",
        "numpy",
        "matplotlib",
    ]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_").split(">=")[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

install_deps()

# Launch streamlit
if __name__ == "__main__":
    os.system("streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0")
