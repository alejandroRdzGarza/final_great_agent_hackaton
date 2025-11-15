# agents/base_agent.py
import uuid
import datetime
import os
from langsmith import Client as LangSmithClient
from holistic_ai_bedrock import get_chat_model

class BaseAgent:
    def __init__(self, name, model_id="claude-3-5-sonnet", temperature=0.0):
        self.name = name
        self.model_id = model_id
        self.temperature = temperature

        # 🚀 Load Holistic AI Bedrock Proxy model
        # (this replaces ANY boto3 use)
        print(f"Loading Bedrock proxy model: {model_id}")
        self.model = get_chat_model(model_id, temperature=temperature)

        # LangSmith
        self.langsmith = LangSmithClient()

        # Agent memory
        self.memory = []

    def call_model(self, prompt):
        """Call the LLM using Holistic AI Bedrock Proxy."""
        result = self.model.invoke(prompt)
        output_text = result.content if hasattr(result, "content") else str(result)

        # Store memory
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.memory.append({
            "timestamp": timestamp,
            "prompt": prompt,
            "response": output_text
        })

        # LangSmith tracing
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
            try:
                self.langsmith.create_run(
                    run_type="llm",
                    name=f"{self.name}_call",
                    inputs={"prompt": prompt},
                    outputs={"response": output_text},
                )
            except Exception as e:
                print(f"⚠️ LangSmith trace failed: {e}")

        return output_text
