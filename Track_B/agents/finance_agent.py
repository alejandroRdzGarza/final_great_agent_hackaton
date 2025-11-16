from .base_agent import BaseAgent
from .advanced_agent import AdvancedAgent
from utils.message_bus import Message

class FinancialAnalystAgent(AdvancedAgent):
    """Financial analysis specialist"""
    
    def __init__(self):
        super().__init__(
            name="FinancialAnalyst",
            role="Financial Assessment Specialist",
            model_id="amazon.nova-micro-v1:0"
        )
    
    def handle_request(self, message: Message) -> dict:
        """Handle financial analysis request"""
        
        claim_data = message.content.get("claim", {})
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   💰 Analyzing costs for {claim_id}")
        
        prompt = f"""You are a financial analyst. Analyze this insurance claim:

CLAIM DATA:
- Type: {claim_data.get('claim_type')}
- Requested Amount: ${claim_data.get('claim_amount'):,.2f}
- Description: {claim_data.get('description')}

Provide:
1. Recommended payout amount
2. Cost breakdown
3. Justification

Respond in JSON format.
"""
        
        response = self.call_model(prompt, thread_id=message.thread_id)
        
        self.memory.set(
            f"financial_analysis_{claim_id}",
            response,
            "Financial analysis completed",
            thread_id=message.thread_id
        )
        
        return {
            "agent": self.name,
            "analysis": response,
            "claim_id": claim_id,
            "status": "completed"
        }
