# agents/risk_analyst.py
from .base_agent import BaseAgent
from .advanced_agent import AdvancedAgent
from utils.message_bus import Message

class RiskAnalystAgent(AdvancedAgent):
    """Risk analysis specialist"""
    
    def __init__(self):
        super().__init__(
            name="RiskAnalyst",
            role="Risk Assessment Specialist",
            model_id="amazon.nova-micro-v1:0"
        )
    
    def handle_request(self, message: Message) -> dict:
        """Handle risk assessment request"""
        
        claim_data = message.content.get("claim", {})
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   🎲 Analyzing risk for {claim_id}")
        
        # Build risk assessment prompt
        prompt = f"""You are a risk analyst. Assess this insurance claim for risk:

CLAIM DATA:
- Type: {claim_data.get('claim_type')}
- Amount: ${claim_data.get('claim_amount'):,.2f}
- Description: {claim_data.get('description')}

Provide:
1. Risk score (0-1, where 1 is highest risk)
2. Risk factors identified
3. Recommendation (low_risk/medium_risk/high_risk)

Respond in JSON format.
"""
        
        response = self.call_model(prompt, thread_id=message.thread_id)
        
        # Store in memory
        self.memory.set(
            f"risk_assessment_{claim_id}",
            response,
            "Risk assessment completed",
            thread_id=message.thread_id
        )
        
        return {
            "agent": self.name,
            "analysis": response,
            "claim_id": claim_id,
            "status": "completed"
        }
