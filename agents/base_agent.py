# agents/base_agent.py
import boto3
import uuid
import datetime
from langsmith import Client as LangSmithClient

class BaseAgent:
    def __init__(self, name, model_id="anthropic.claude-v2", temperature=0.0):
        """
        model_id: Bedrock model ID, e.g., 'anthropic.claude-v2', 'amazon.titan-1', 'mistral-small'
        """
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        
        # AWS Bedrock client
        self.bedrock_client = boto3.client("bedrock")  
        
        # LangSmith client for tracing
        self.langsmith = LangSmithClient()
        
        # Local memory
        self.memory = []

    def log_memory(self, input_text, output_text, tool_used=None):
        """Logs agent decisions locally and to LangSmith"""
        event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": self.name,
            "input": input_text,
            "output": output_text,
            "tool_used": tool_used,
            "event_id": str(uuid.uuid4())
        }
        self.memory.append(event)
        self.langsmith.log_event(event)
        return event

    def call_model(self, prompt):
        """Call AWS Bedrock LLM"""
        response = self.bedrock_client.invoke_model(
            ModelId=self.model_id,
            Body=prompt.encode("utf-8"),
            ContentType="text/plain",
            Accept="text/plain",
            # You can add more parameters like MaxTokens if needed
        )
        output_text = response["Body"].read().decode("utf-8")
        return self.log_memory(prompt, output_text)
