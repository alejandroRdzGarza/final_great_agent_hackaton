# agents/investigator_agent.py
import os
import json
from typing import Optional, Dict
from agents.advanced_agent import AdvancedAgent
from tools.valyu_search import valyu_search
from utils.message_bus import Message
import datetime

class InvestigatorAgent(AdvancedAgent):
    """Investigator with conversation memory"""
    
    def __init__(self, name="Investigator", model_id="claude-3-5-sonnet"):
        super().__init__(
            name=name,
            role="Investigation Specialist",
            model_id=model_id
        )
        self.search_tool = valyu_search
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Handle investigation request with full context"""
        
        content = message.content
        request_type = content.get("type", "investigate")
        claim_data = content.get("claim", {})
        claim_id = content.get("claim_id", "UNKNOWN")
        
        print(f"   🔍 Investigating {claim_id}")
        
        # Build prompt with awareness of conversation
        prompt = f"""You are an insurance claims investigator.

TASK: Investigate claim {claim_id}

CLAIM DATA:
{json.dumps(claim_data, indent=2)}

Based on previous conversations and this claim data:
1. What evidence have we gathered so far?
2. What additional investigation is needed?
3. What's your assessment?

Provide a structured investigation report."""
        
        # Call model WITH conversation context
        response = self.call_model(
            prompt, 
            thread_id=message.thread_id,
            include_conversation=True  # KEY: Include full conversation
        )
        
        # Store in memory
        thread_id = message.thread_id or "unknown"
        self.memory.set(
            f"investigation_{claim_id}",
            response,
            "Investigation completed",
            thread_id=thread_id
        )
        
        return {
            "agent": self.name,
            "investigation": response,
            "claim_id": claim_id,
            "status": "completed"
        }