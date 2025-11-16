# utils/report_generator.py
from typing import Dict, List, Any
from datetime import datetime
import json

class ReportGenerator:
    """
    Generates professional reports for hackathon submission.
    
    Creates a 1-2 page company-style report with:
    - Executive summary
    - Team contributions
    - Trace graph/timeline
    - Key risks identified
    - Internal debates summary
    - Final decision
    """
    
    def __init__(self, project_name: str, team_name: str):
        self.project_name = project_name
        self.team_name = team_name
    
    def generate_report(
        self,
        claim_data: Dict,
        agents_used: List[str],
        auditor_summary: Dict,
        debate_summary: Dict,
        trace_data: Dict,
        final_decision: str
    ) -> str:
        """
        Generate complete professional report.
        
        Returns: Markdown-formatted report string
        """
        
        report = f"""# {self.project_name}
## Glass Box AI - Transparent Insurance Claims Processing
### Team: {self.team_name}
### Date: {datetime.now().strftime("%B %d, %Y")}

---

## Executive Summary

This report details the transparent, auditable decision-making process for insurance claim #{claim_data.get('claim_id', 'N/A')}.

**Claim Details:**
- Claimant: {claim_data.get('claimant_name', 'N/A')}
- Type: {claim_data.get('claim_type', 'N/A')}
- Amount: ${claim_data.get('claim_amount', 0):,.2f}
- Status: {claim_data.get('status', 'N/A')}

**Processing Summary:**
- Agents Involved: {len(agents_used)}
- Total Deliberation Rounds: {debate_summary.get('total_rounds', 0)}
- Consensus Reached: {debate_summary.get('consensus_reached', False)}
- Average Transparency Score: {auditor_summary.get('avg_transparency_score', 0):.2f}/1.0

---

## Team Contributions

### Agent Roles & Responsibilities

"""
        
        # Add agent descriptions
        for agent_name in agents_used:
            report += f"**{agent_name}**: "
            if "Claims" in agent_name:
                report += "Initial claim validation and documentation review\n"
            elif "Risk" in agent_name:
                report += "Risk assessment and fraud detection analysis\n"
            elif "Financial" in agent_name:
                report += "Cost analysis and payout calculations\n"
            elif "Auditor" in agent_name:
                report += "Transparency auditing and reasoning verification\n"
            elif "Planner" in agent_name:
                report += "Task coordination and workflow orchestration\n"
            else:
                report += "Specialized processing and analysis\n"
        
        report += f"""

---

## Decision Process Timeline

### Phase 1: Initial Assessment
{self._format_trace_timeline(trace_data)}

### Phase 2: Multi-Agent Deliberation
{self._format_debate_summary(debate_summary)}

### Phase 3: Transparency Audit
{self._format_audit_summary(auditor_summary)}

---

## Key Risks Identified

"""
        
        # Extract risks from audit
        common_issues = auditor_summary.get('common_issues', [])
        if common_issues:
            for i, issue in enumerate(common_issues[:5], 1):
                report += f"{i}. {issue}\n"
        else:
            report += "No significant risks identified during audit.\n"
        
        report += f"""

---

## Internal Debates & Reasoning

### Debate Topic: {debate_summary.get('topic', 'Claim Approval Decision')}

**Participants**: {', '.join(debate_summary.get('participants', []))}

**Key Discussion Points:**
"""
        
        # Add debate positions
        if 'rounds' in debate_summary and debate_summary['rounds']:
            last_round = debate_summary['rounds'][-1]
            for position in last_round.get('positions', []):
                report += f"\n- **{position['agent']}** ({position['position']}): {position['reasoning'][:200]}...\n"
        
        report += f"""

**Consensus**: {"✅ Reached" if debate_summary.get('consensus_reached') else "❌ Not reached"}

---

## Final Decision

{final_decision}

---

## Transparency Metrics

| Metric | Score |
|--------|-------|
| Average Transparency | {auditor_summary.get('avg_transparency_score', 0):.2f}/1.0 |
| Shortcut Risk | {auditor_summary.get('avg_shortcut_risk', 0):.2f}/1.0 |
| Total Audits Performed | {auditor_summary.get('total_audits', 0)} |
| Flags Raised | {auditor_summary.get('total_flags', 0)} |

---

## Trace Visualization

View complete execution trace in LangSmith:
- Project: {trace_data.get('project', 'insurance-multi-agent')}
- Thread ID: {trace_data.get('thread_id', 'N/A')}
- URL: https://smith.langchain.com

---

## Conclusion

This transparent, auditable process demonstrates the Glass Box approach to AI decision-making in insurance. Every step is documented, every decision is reasoned, and every risk is identified.

**Key Achievements:**
✅ Full traceability of all agent interactions
✅ Multi-perspective deliberation on complex decisions  
✅ Automated transparency auditing
✅ Human-interpretable reasoning chains
✅ Real-time visibility through LangSmith

---

*Report generated automatically by Glass Box AI System*  
*{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC*
"""
        
        return report
    
    def _format_trace_timeline(self, trace_data: Dict) -> str:
        """Format trace timeline"""
        timeline = ""
        
        if 'events' in trace_data:
            for i, event in enumerate(trace_data['events'][:10], 1):
                timeline += f"{i}. **{event.get('agent', 'Unknown')}**: {event.get('action', 'No action')} ({event.get('timestamp', 'N/A')})\n"
        else:
            timeline += "Timeline visualization available in LangSmith dashboard.\n"
        
        return timeline
    
    def _format_debate_summary(self, debate_summary: Dict) -> str:
        """Format debate summary"""
        if not debate_summary.get('rounds'):
            return "No debate rounds conducted.\n"
        
        summary = f"Total Rounds: {len(debate_summary['rounds'])}\n\n"
        
        for round_data in debate_summary['rounds']:
            summary += f"**Round {round_data['round_number']}:**\n"
            for pos in round_data.get('positions', []):
                summary += f"- {pos['agent']}: {pos['position']}\n"
            summary += "\n"
        
        return summary
    
    def _format_audit_summary(self, audit_summary: Dict) -> str:
        """Format audit summary"""
        return f"""
- **Transparency Score**: {audit_summary.get('avg_transparency_score', 0):.2f}/1.0
- **Shortcut Risk**: {audit_summary.get('avg_shortcut_risk', 0):.2f}/1.0
- **Total Flags**: {audit_summary.get('total_flags', 0)}
- **Audits Performed**: {audit_summary.get('total_audits', 0)}
"""
    
    def save_report(self, report: str, filename: str = "glass_box_report.md"):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {filename}")