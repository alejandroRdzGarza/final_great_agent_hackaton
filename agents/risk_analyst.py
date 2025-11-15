# agents/risk_analyst.py
from .base_agent import BaseAgent

class RiskAnalystAgeng(BaseAgent):
    def __init__(self):
        super().__init__(name="RiskAnalyst")

    def assess_risk(self, claim_event):
        claim_text = claim_event["output"]
        prompt = f"Analyze risk for the following claim and estimate liability percentage:\n{claim_text}"
        event = self.call_model(prompt)
        return event
