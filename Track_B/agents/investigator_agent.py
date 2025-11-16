import os
import json
from typing import Optional, Dict
from agents.advanced_agent import AdvancedAgent
from valyu_search import valyu_search  # Now using the working version
from utils.message_bus import Message
import datetime

class InvestigatorAgent(AdvancedAgent):
    """Investigator with conversation memory and Valyu search capabilities"""
    
    def __init__(self, name="Investigator", model_id="claude-3-5-sonnet"):
        super().__init__(
            name=name,
            role="Investigation Specialist",
            model_id=model_id
        )
        self.search_tool = valyu_search
        self.tools = [self.search_tool]  # Make sure tools are registered
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Handle investigation request with full context and search capabilities"""
        
        content = message.content
        request_type = content.get("type", "investigate")
        claim_data = content.get("claim", {})
        claim_id = content.get("claim_id", "UNKNOWN")
        
        print(f"   🔍 Investigating {claim_id}")
        
        # Build investigation context
        investigation_context = self._build_investigation_context(claim_data, claim_id)
        
        # Call model WITH conversation context and search capabilities
        response = self.call_model(
            investigation_context, 
            thread_id=message.thread_id,
            include_conversation=True,  # KEY: Include full conversation
            tools=self.tools  # Enable search tool
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
    
    def _build_investigation_context(self, claim_data: Dict, claim_id: str) -> str:
        """Build comprehensive investigation prompt"""
        
        # Extract key claim information for search queries
        claimant_name = claim_data.get("claimant_name", "Unknown")
        claim_type = claim_data.get("claim_type", "Unknown")
        claim_amount = claim_data.get("claim_amount", 0)
        description = claim_data.get("description", "")
        
        # Create search-friendly queries based on claim data
        search_queries = [
            f"insurance claim fraud detection {claim_type}",
            f"homeowners insurance business use exclusion",
            f"insurance material misrepresentation cases",
            f"{claim_type} claim investigation techniques"
        ]
        
        prompt = f"""You are an insurance claims investigator.

TASK: Investigate claim {claim_id}

CLAIM DATA:
- Claimant: {claimant_name}
- Type: {claim_type}
- Amount: ${claim_amount:,.2f}
- Description: {description[:500]}...

INVESTIGATION STEPS:

1. **Evidence Review**: Analyze the provided claim data for inconsistencies
2. **External Research**: Use the search tool to find:
   - Similar fraud cases
   - Legal precedents for business exclusions
   - Industry best practices for {claim_type} claims
3. **Risk Assessment**: Evaluate the likelihood of fraud or misrepresentation
4. **Recommendations**: Suggest next steps for investigation

SEARCH QUERIES TO CONSIDER:
{chr(10).join(f"- {query}" for query in search_queries)}

Provide a structured investigation report with:
- Key findings from claim analysis
- Relevant external information found via search
- Risk assessment
- Recommended actions
"""

        return prompt
    
    def perform_targeted_search(self, query: str) -> str:
        """Perform targeted search using Valyu"""
        try:
            print(f"   🔎 Performing targeted search: {query}")
            return self.search_tool.invoke({"query": query})
        except Exception as e:
            return f"Search failed: {str(e)}"