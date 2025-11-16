# agents/base_agent.py
import uuid
import datetime
import os
from holistic_ai_bedrock import get_chat_model
from langchain_core.runnables import RunnableConfig

class BaseAgent:
    def __init__(self, name, model_id="claude-3-5-sonnet", temperature=0.0):
        self.name = name
        self.model_id = model_id
        self.temperature = temperature

        print(f"🤖 Initializing agent: {name}")
        print(f"   Model: {model_id}")
        
        # Load model - automatically traces when LANGCHAIN_TRACING_V2=true
        self.model = get_chat_model(model_id, temperature=temperature)
        print(f"   ✅ Model loaded with automatic tracing")
        
        # Check tracing
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            project = os.getenv("LANGCHAIN_PROJECT", "default")
            print(f"   📊 Tracing to: {project}")
        else:
            print(f"   ⚠️  Tracing disabled")

        self.memory = []

    def call_model(self, prompt, thread_id=None, **kwargs):
        """
        Call model with LangSmith thread support.
        
        CRITICAL: Pass thread_id in metadata with key 'thread_id', 'session_id', or 'conversation_id'
        This groups all traces into a visual thread in LangSmith UI.
        """
        
        print(f"\n🤖 [{self.name}] Calling model...")
        if thread_id:
            print(f"   Thread ID: {thread_id}")
        
        # Create config with thread_id in metadata
        # This is what creates the thread in LangSmith!
        config = RunnableConfig(
            metadata={
                "thread_id": str(thread_id),  # One of: thread_id, session_id, conversation_id
                "agent_name": self.name,
            },
            tags=[self.name, "agent_call"]
        )
        
        # Invoke with config - automatically creates trace with thread_id
        result = self.model.invoke(prompt, config=config)
        output_text = result.content if hasattr(result, "content") else str(result)
        
        print(f"   ✅ Response: {len(output_text)} chars")
        
        # Store in memory
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.memory.append({
            "timestamp": timestamp,
            "prompt": prompt,
            "response": output_text,
            "thread_id": thread_id
        })

        return output_text