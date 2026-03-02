from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_fewshot_messages(jsonl_path: str, limit: int = 6) -> List[Dict[str, str]]:
    """
    Loads few-shot examples from JSONL where each line is a message dict:
      e.g. {"role": "user"|"assistant", "content": "..."}

    Arguments:
        limit : int
            Maximum number of pairs of user-assistant messages to load 
            (in the order specified in the example data).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []

    messages: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages.append(json.loads(line))

            if len(messages) // 2 >= limit:
                break
    
    return messages

# TO-DO: randomize selection of examples