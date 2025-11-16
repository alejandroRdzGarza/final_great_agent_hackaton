# org/tasks.py
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

@dataclass
class InsuranceClaim:
    claim_id: str
    claimant_name: str
    claim_type: str
    claim_amount: float
    description: str
    status: str = "Pending"


class Task:
    def __init__(self, description: str, assigned_agent=None, priority: int = 1, claim=None):
        self.id = str(uuid.uuid4())
        self.description = description
        self.assigned_agent = assigned_agent
        self.status = "pending"  # pending, in_progress, done
        self.created_at = datetime.now(timezone.utc)
        self.completed_at = None
        self.priority = priority
        self.dependencies = []
        self.output = None
        self.claim = claim

    def assign_agent(self, agent):
        self.assigned_agent = agent

    def complete(self, output):
        self.status = "done"
        self.completed_at = datetime.now(timezone.utc)
        self.output = output
