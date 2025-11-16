# agents/base_agent_comms.py
from agents.base_agent import BaseAgent
import datetime
import os

class OrganizationalAgent(BaseAgent):
    def __init__(self, name, model_id="amazon.titan-1"):
        super().__init__(name, model_id)
        self.inbox = []

    def receive_message(self, message):
        """Receive a message and store in memory and inbox"""
        self.inbox.append(message)

        # Log locally
        timestamp = datetime.datetime.now().isoformat()
        self.memory.append({
            "timestamp": timestamp,
            "event": "receive_message",
            "from": message.sender.name,
            "content": message.content
        })

        # Optional: LangSmith tracing
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            try:
                self.langsmith.create_run(
                    name=f"{self.name}_receive_message",
                    run_type="chain",
                    inputs={"from": message.sender.name, "content": message.content},
                    outputs={"status": "received"},
                    metadata={"agent": self.name}
                )
            except Exception as e:
                print(f"⚠️ LangSmith trace failed: {e}")

    def send_message(self, receiver, content, hub, task_id=None):
        """Send a message via the communication hub"""
        # Log locally
        timestamp = datetime.datetime.now().isoformat()
        self.memory.append({
            "timestamp": timestamp,
            "event": "send_message",
            "to": receiver.name,
            "content": content
        })

        # Optional: LangSmith tracing
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            try:
                self.langsmith.create_run(
                    name=f"{self.name}_send_message",
                    run_type="chain",
                    inputs={"to": receiver.name, "content": content},
                    outputs={"status": "sent"},
                    metadata={"agent": self.name}
                )
            except Exception as e:
                print(f"⚠️ LangSmith trace failed: {e}")

        # Actually send the message
        return hub.send_message(self, receiver, content, task_id)
    
    def process_next_message(self, hub):
        if not self.inbox:
            return
        message = self.inbox.pop(0)

        # Generate LLM response
        prompt = f"Agent {self.name}, you received this message:\n{message.content}\nRespond appropriately."
        
        # Pass thread_id from the message
        thread_id = getattr(message, "thread_id", None)
        response = self.call_model(prompt, thread_id=thread_id)

        # Send response back, preserve the thread_id
        hub.send_message(self, message.sender, response, task_id=thread_id)


