"""
Logger minimal : console + fichier optionnel.
"""
import os
from datetime import datetime

class SimpleLogger:
    def __init__(self, logdir: str = "results/logs"):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logfile = os.path.join(logdir, f"log_{now}.txt")

    def info(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[INFO] {timestamp} {msg}"
        print(line)
        with open(self.logfile, "a") as f:
            f.write(line + "\n")

    def debug(self, msg: str):
        print("[DEBUG]", msg)
