# org/tasks.py
import uuid
from datetime import datetime, timezone

class Task:
    def __init__(self, description: str, assigned_agent=None, priority: int = 1):
        self.id = str(uuid.uuid4())
        self.description = description
        self.assigned_agent = assigned_agent
        self.status = "pending"  # pending, in_progress, done
        self.created_at = datetime.now(timezone.utc)
        self.completed_at = None
        self.priority = priority
        self.dependencies = []
        self.output = None

    def assign_agent(self, agent):
        self.assigned_agent = agent

    def complete(self, output):
        self.status = "done"
        self.completed_at = datetime.now(timezone.utc)
        self.output = output
