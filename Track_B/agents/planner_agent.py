# agents/planner_agent.py
from agents.base_agent import BaseAgent
import json
from langsmith import uuid7

class PlannerAgent(BaseAgent):
    def __init__(self, name, model_id="claude-3-5-sonnet", communication_hub=None, agents=None):
        super().__init__(name=name, model_id=model_id)
        self.communication_hub = communication_hub
        self.agents = agents or []  # list of agent instances

    def plan(self, task_input):
        if isinstance(task_input, dict):
            task_desc = task_input.get("description", "")
            claim = task_input.get("claim", None)
        else:
            task_desc = str(task_input)
            claim = None

        prompt = f"""
    You are a planner agent. You receive a task:
    {task_desc}

    Available agents: {', '.join([a.name for a in self.agents])}

    Break the task into subtasks and assign each subtask to the most appropriate agent.
    Include the claim data in each subtask if needed.
    Respond in JSON format:
    [
    {{"agent": "AgentName", "subtask": "Description"}},
    ...
    ]
    """
        result = self.model.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)
        subtasks = self._parse_response(response)

        # Attach claim to each subtask
        for st in subtasks:
            st["claim"] = claim

        self._dispatch_subtasks(subtasks)


    def _parse_response(self, response):
        try:
            subtasks = json.loads(response)
            return subtasks
        except Exception:
            # fallback: simple line parser
            tasks = []
            for line in response.splitlines():
                if ":" in line:
                    agent_name, subtask = line.split(":", 1)
                    tasks.append({"agent": agent_name.strip(), "subtask": subtask.strip()})
            return tasks

    def _dispatch_subtasks(self, subtasks):
        if not self.communication_hub:
            raise ValueError("Communication hub is required to dispatch subtasks.")

        # Create one thread_id per task if not already present
        thread_id = uuid7()

        for t in subtasks:
            agent = next((a for a in self.agents if a.name == t["agent"]), None)
            if agent:
                content = t["subtask"]
                
                # If the task has an associated claim, append claim info
                claim = t.get("claim", None)
                if claim:
                    content += f"\n\nClaim details:\n{claim}"

                self.communication_hub.send_message(
                    sender=self,
                    receiver=agent,
                    content=content,
                    task_id=thread_id  # Attach thread_id to all subtasks
                )


                
    def receive_message(self, message):
        """Receive a message just like other agents (optional logging)."""
        # Store in memory if needed
        self.memory.append({
            "timestamp": message.timestamp.isoformat(),
            "event": "receive_message",
            "from": message.sender.name,
            "content": message.content
        })
        # Optional: print in real-time
        print(f"{message.timestamp} | {message.sender.name} -> {self.name}: {message.content}")
