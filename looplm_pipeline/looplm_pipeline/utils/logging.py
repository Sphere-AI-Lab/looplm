import json
import os
from typing import Dict

from .dist import is_main_process
from .io import ensure_dir


class JsonlLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path) or ".")

    def log(self, payload: Dict, also_print: bool = True) -> None:
        if not is_main_process():
            return
        line = json.dumps(payload, ensure_ascii=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print:
            print(line, flush=True)
