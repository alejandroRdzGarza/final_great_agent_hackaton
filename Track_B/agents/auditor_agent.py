# agents/auditor_agent.py
from agents.base_agent import BaseAgent
from org.schemas import TransparencyReport, ReasoningStep
from typing import List, Dict
import json
from .base_agent import BaseAgent
from .advanced_agent import AdvancedAgent
from utils.message_bus import Message
import datetime
from datetime import timezone

class AuditorAgent(AdvancedAgent):
    """Transparency auditor"""
    
    def __init__(self):
        super().__init__(
            name="TransparencyAuditor",
            role="Transparency & Compliance Auditor",
            model_id="amazon.nova-micro-v1:0"
        )
        self.audits = []
    
    # agents/auditor_agent.py

    def handle_request(self, message: Message) -> dict:
        """Handle audit request with full conversation context"""
        
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   🔍 Auditing transparency for {claim_id}")
        
        # Get full conversation context from this thread
        conversation_context = self._get_conversation_context(message.thread_id)
        
        prompt = f"""You are a transparency auditor reviewing this insurance claim processing.

    {conversation_context}

    Review the ENTIRE conversation above and rate:
    1. Transparency Score (0-1): How clear and documented is the reasoning?
    2. Reasoning Completeness (0-1): Are all steps explained?
    3. Risk of Shortcuts (0-1): Did agents skip steps or jump to conclusions?
    4. Issues Found: List any transparency problems

    IMPORTANT: Review ALL the agent responses in the conversation history.

    Respond in JSON format:
    {{
    "transparency_score": 0.0-1.0,
    "reasoning_completeness": 0.0-1.0,
    "risk_of_shortcuts": 0.0-1.0,
    "issues_found": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
    }}
    """
        
        response = self.call_model(prompt, thread_id=message.thread_id, include_conversation=True)
        
        return {
            "agent": self.name,
            "audit": response,
            "claim_id": claim_id,
            "status": "completed"
        }

    
    def audit_decision(self, agent_decision: Dict, thread_id: str | None = None) -> TransparencyReport:
        """
        Audit an agent's decision for transparency.
        
        Returns a TransparencyReport with scores and flags.
        """
        
        print(f"\n🔍 [{self.name}] Auditing decision from {agent_decision.get('agent_name')}")
        
        # Build audit prompt
        prompt = f"""You are an AI Auditor. Analyze this agent's decision for transparency and reasoning quality.

AGENT DECISION:
{json.dumps(agent_decision, indent=2)}

AUDIT CRITERIA:
1. Reasoning Completeness (0-1): Are all steps explained?
2. Risk of Shortcut (0-1): Did the agent skip reasoning or jump to conclusions?
3. Tool Usage: Are all tools documented?
4. Citations: Are sources properly cited?
5. Consistency: Do the steps logically flow?
6. Hallucination Risk: Any unsupported claims?

Respond in JSON format:
{{
  "transparency_score": 0.0-1.0,
  "risk_of_shortcut_score": 0.0-1.0,
  "flags": ["flag1", "flag2", ...],
  "recommendations": ["rec1", "rec2", ...],
  "reasoning": "Brief explanation of scores"
}}
"""
        
        # Call model with thread_id for tracing
        response = self.call_model(prompt, thread_id=thread_id)
        
        # Parse response
        try:
            audit_result = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            audit_result = {
                "transparency_score": 0.5,
                "risk_of_shortcut_score": 0.5,
                "flags": ["Failed to parse audit response"],
                "recommendations": ["Retry audit with clearer output format"],
                "reasoning": "Audit parsing failed"
            }
        
        # Create transparency report
        report = TransparencyReport(
            agent_name=agent_decision.get('agent_name', 'Unknown'),
            transparency_score=audit_result.get('transparency_score', 0.5),
            risk_of_shortcut_score=audit_result.get('risk_of_shortcut_score', 0.5),
            reasoning_timeline=[],  # Extract from agent_decision steps
            flags=audit_result.get('flags', []),
            recommendations=audit_result.get('recommendations', [])
        )
        
        # Store audit
        self.audits.append(report)
        
        print(f"   📊 Transparency Score: {report.transparency_score:.2f}")
        print(f"   ⚠️  Shortcut Risk: {report.risk_of_shortcut_score:.2f}")
        print(f"   🚩 Flags: {len(report.flags)}")
        
        return report
    
    def generate_audit_summary(self) -> Dict:
        """Generate summary of all audits"""
        
        if not self.audits:
            return {"message": "No audits performed yet"}
        
        avg_transparency = sum(a.transparency_score for a in self.audits) / len(self.audits)
        avg_shortcut_risk = sum(a.risk_of_shortcut_score for a in self.audits) / len(self.audits)
        
        all_flags = []
        for audit in self.audits:
            all_flags.extend(audit.flags)
        
        return {
            "total_audits": len(self.audits),
            "avg_transparency_score": avg_transparency,
            "avg_shortcut_risk": avg_shortcut_risk,
            "total_flags": len(all_flags),
            "common_issues": list(set(all_flags)),
            "audits": [a.dict() for a in self.audits]
        }