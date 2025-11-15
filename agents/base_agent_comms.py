# agents/base_agent_communication.py
from agents.base_agent import BaseAgent

class OrganizationalAgent(BaseAgent):
    def __init__(self, name, model_id="amazon.titan-1"):
        super().__init__(name, model_id)
        self.inbox = []

    def receive_message(self, message):
        self.inbox.append(message)
        self.log_memory(f"Received message from {message.sender.name}", message.content)

    def send_message(self, receiver, content, hub, task_id=None):
        return hub.send_message(self, receiver, content, task_id)
