# agents/decision_maker_agent.py

from agents.advanced_agent import AdvancedAgent
from utils.message_bus import Message, MessageType
from typing import Dict, Optional
import json

class DecisionMakerAgent(AdvancedAgent):
    """
    Final decision maker that synthesizes all agent analyses into a 
    binding decision with full justification.
    """
    
    def __init__(self):
        super().__init__(
            name="DecisionMaker",
            role="Chief Claims Officer",
            model_id="amazon.nova-micro-v1:0"
        )
        self.final_decisions = {}
    
    # agents/decision_maker_agent.py - Update handle_request

    def handle_request(self, message: Message) -> Optional[Dict]:
        """
        Synthesize all agent analyses into final decision.
        """
        
        claim_id = message.content.get("claim_id", "UNKNOWN")
        claim_data = message.content.get("claim", {})
        
        print(f"   ⚖️  Making final decision for {claim_id}")
        
        # Get FORMATTED conversation context with all analyses
        thread_id = message.thread_id or "unknown"
        conversation_context = self._get_conversation_context(
            thread_id, 
            format_for_decision=True  # NEW: Use decision-friendly format
        )
        
        prompt = f"""You are the Chief Claims Officer making the FINAL BINDING DECISION on this insurance claim.

    CLAIM INFORMATION:
    - ID: {claim_id}
    - Claimant: {claim_data.get('claimant_name')}
    - Type: {claim_data.get('claim_type')}
    - Requested Amount: ${claim_data.get('claim_amount', 0):,.2f}
    - Description: {claim_data.get('description', 'N/A')}

    {conversation_context}

    YOU HAVE ALL THE TEAM ANALYSES ABOVE. NOW MAKE YOUR DECISION:

    Respond in JSON format:
    {{
    "decision": "APPROVE" or "DENY",
    "final_payout_amount": 0.00,
    "justification": "Specific references to team findings",
    "conditions": ["condition1", "condition2"],
    "confidence": 0.0-1.0,
    "summary": "One sentence decision summary"
    }}

    MAKE THE FINAL BINDING DECISION NOW based on all analyses above.
    """
        
        response = self.call_model(
            prompt, 
            thread_id=message.thread_id,
            include_conversation=False  # We already formatted it ourselves
        )
        
        # Parse JSON response
        try:
            decision_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            decision_data = {
                "decision": "APPROVE",
                "final_payout_amount": claim_data.get('claim_amount', 0) * 0.9,
                "justification": response[:500],  # Use first 500 chars as justification
                "conditions": [],
                "confidence": 0.7,
                "summary": "Decision made based on team analysis"
            }
        
        # Store decision
        self.final_decisions[claim_id] = decision_data
        
        # Store in memory
        thread_id = message.thread_id or "unknown"
        self.memory.set(
            f"final_decision_{claim_id}",
            decision_data,
            "Final binding decision made",
            thread_id=thread_id
        )
        
        print(f"   ✅ Decision: {decision_data.get('decision')} - ${decision_data.get('final_payout_amount', 0):,.2f}")
        
        return {
            "agent": self.name,
            "decision": decision_data,
            "claim_id": claim_id,
            "status": "completed"
        }