import uuid
from typing import Any, Optional

STAGES = ["etl", "stats", "model", "evaluate", "scoring"]
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}


class SessionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.session_id = str(uuid.uuid4())
        self.current_stage: Optional[str] = None
        self.stage_status: dict[str, str] = {s: "pending" for s in STAGES}
        self.stage_outputs: dict[str, Any] = {}
        self.confirmed_outputs: dict[str, Any] = {}
        self.dataset_path: Optional[str] = None
        self.dataset_summary: Optional[dict] = None
        self.agent_logs: dict[str, str] = {}

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "current_stage": self.current_stage,
            "stage_status": self.stage_status,
            "confirmed_outputs": self.confirmed_outputs,
        }

    def can_run_stage(self, stage: str) -> bool:
        idx = STAGE_ORDER[stage]
        if idx == 0:
            if self.dataset_path is not None:
                return True
            # Allow if dataset already exists on disk
            from pathlib import Path
            default_path = next(iter(sorted((Path(__file__).resolve().parent.parent / "data").glob("*.csv"))), None)
            if default_path and default_path.exists():
                self.dataset_path = str(default_path)
                return True
            return False
        prev = STAGES[idx - 1]
        return self.stage_status[prev] in ("confirmed", "complete")


session = SessionState()
