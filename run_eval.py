import subprocess
import time
import sys

from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
STARTUP_WAIT_SECONDS = 5

# Start uvicorn server
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "domain_qa.app.main:app", "--reload"],
)

# This file's directory = project root (if script is in root)
PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_PATH = PROJECT_ROOT / "reports" / "eval.txt"

try:
    # Wait for server to start
    time.sleep(STARTUP_WAIT_SECONDS)

    # Run evaluation
    subprocess.run(
        [
            sys.executable,
            "-m",
            "domain_qa.eval.run",
            "--base-url",
            f"http://{HOST}:{PORT}",
            "--report-path",
            str(REPORT_PATH),
        ],
        check=True,
    )
finally:
    server.terminate()
    server.wait()