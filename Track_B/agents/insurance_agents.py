# agents/insurance_investigator.py
from agents.advanced_agent import AdvancedAgent
from utils.message_bus import Message
from typing import Optional, Dict

# Load the core behavior prompt (shared across all agents)
CORE_INSURANCE_PROTOCOL = """
AI Agent Behaviour: Core Insurance Protocol -----

Your identity is that of a professional, ethical, and expert insurance agent.
Your actions and decisions are governed by the following core principles.

1. Primary Mandate: Act in Utmost Good Faith  
This is your non-negotiable foundation. You must act with absolute honesty, integrity,
and fairness in all interactions. You have a duty to both the policyholder and the
company, and you must balance their interests equitably.

2. Core Value: Professionalism and Fairness  
Professionalism: You are an expert. Your tone is courteous, clear, and empathetic.
You do not use confusing jargon. You are patient and reliable.  
Fairness: You investigate all matters objectively. You must not misrepresent facts
or policy provisions. You cannot conceal benefits. Your goal is to reach a correct
and just outcome based on the facts and the policy, free from bias.  
Empathy: You acknowledge the human element of insurance. When a user is in a
stressful situation (like a claim), you are supportive and clear, but remain objective
in your final decision-making.

3. Guiding Principle: Total Observability  
You must operate with 100% transparency. Every decision, conclusion, or recommendation
you make must be justified.  
Always Justify: You will always provide a clear and simple explanation for your answers,
decisions, or recommendations.  
Cite Your Sources: When you make a decision, you must state the reason for it. This reason
must be based on specific, observable facts (e.g., "the information you provided") or specific
rules (e.g., "the terms of the policy contract").  
Explain "Why": Never just give an answer; explain why it is the answer.

4. Operational Directives  
Assume Nothing: Do not make assumptions. If you require a piece of information to make
a decision, you must ask for it.  
Uphold the Contract: The policy is a legal contract. Your analysis must be based on the facts
of the situation as applied to the exact wording of the policy.  
Clarity Above All: You will use plain, easily understood language. It is your job to make complex
insurance topics simple. If you must use a technical term, you must immediately explain it.  
Log Your Reasoning (Internal Directive): Every step in your decision-making process must be
logged and traceable, allowing for a complete audit of your reasoning.
"""


from valyu import Valyu
import os

class SIUInvestigatorAgent(AdvancedAgent):
    """
    Special Investigations Unit (SIU) Investigator with Valyu Search
    
    Focus: Fraud detection, evidence gathering, professional skepticism
    Output: Investigation report with confidence level (0-100%)
    """
    
    def __init__(self, model_id: str = "amazon.nova-micro-v1:0"):
        super().__init__(
            name="SIU_Investigator",
            role="Special Investigations Unit",
            model_id=model_id
        )
        self.investigations = {}
        
        # Initialize Valyu client
        api_key = os.getenv("VALYU_API_KEY")
        if not api_key:
            print("⚠️  VALYU_API_KEY not found - search capabilities disabled")
            self.valyu_client = None
        else:
            try:
                self.valyu_client = Valyu(api_key=api_key)
                print("✅ Valyu search initialized for SIU Investigator")
            except Exception as e:
                print(f"⚠️  Failed to initialize Valyu: {e}")
                self.valyu_client = None
    
    def perform_valyu_search(self, query: str) -> str:
        """Perform search using Valyu and return formatted results"""
        if not self.valyu_client:
            return "Search unavailable - Valyu not configured"
        
        try:
            print(f"   🔍 Valyu searching: {query}")
            response = self.valyu_client.search(query)
            
            if not response or not response.results:
                return "No relevant information found in search."
            
            # Format results for the agent's analysis
            formatted_results = []
            for i, result in enumerate(response.results[:3], 1):  # Top 3 results
                title = getattr(result, 'title', 'Untitled')
                content = getattr(result, 'content', 'No content available')[:300]  # Limit content length
                url = getattr(result, 'url', 'No URL')
                
                formatted_results.append(
                    f"{i}. {title}\n"
                    f"   Source: {url}\n"
                    f"   Content: {content}...\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Handle fraud investigation request with Valyu search integration"""
        
        claim_data = message.content.get("claim", {})
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   🔍 SIU investigating {claim_id}")
        
        # Get any previous context
        conversation_context = self._get_conversation_context(message.thread_id)
        
        # Perform targeted searches based on claim type
        search_results = self._perform_investigation_searches(claim_data)
        
        # Build investigation prompt with search results
        prompt = self._build_investigation_prompt(claim_data, claim_id, conversation_context, search_results)
        
        # Call model
        response = self.call_model(
            prompt,
            thread_id=message.thread_id,
            include_conversation=False  # We included it manually
        )
        
        # Store investigation
        self.investigations[claim_id] = response
        
        # Store in memory
        self.memory.set(
            f"siu_investigation_{claim_id}",
            response,
            "SIU investigation completed with confidence assessment",
            thread_id=message.thread_id
        )
        
        print(f"   ✅ Investigation complete")
        
        return {
            "agent": self.name,
            "investigation": response,
            "claim_id": claim_id,
            "search_queries_used": list(search_results.keys()),
            "status": "completed"
        }
    
    def _perform_investigation_searches(self, claim_data: Dict) -> Dict[str, str]:
        """Perform targeted searches based on claim characteristics"""
        search_results = {}
        
        if not self.valyu_client:
            return search_results
        
        # Extract key claim information for search queries
        claim_type = claim_data.get("claim_type", "").lower()
        description = claim_data.get("description", "").lower()
        claimant_name = claim_data.get("claimant_name", "")
        
        # Define search queries based on claim content
        search_queries = []
        
        # Business vs hobby disputes (common in homeowners claims)
        if any(word in description for word in ['business', 'hobby', 'etsy', 'llc', 'sales']):
            search_queries.extend([
                "insurance business pursuit exclusion homeowners policy",
                "hobby vs business insurance claims court cases",
                "home-based business insurance coverage denial"
            ])
        
        # Fire-related claims
        if 'fire' in description:
            search_queries.extend([
                "fire cause and origin investigation insurance",
                "electrical fire claims homeowners insurance",
                "workshop fire insurance claims business exclusion"
            ])
        
        # Material misrepresentation
        if any(word in description for word in ['misrepresentation', 'conceal', 'disclose', 'lied']):
            search_queries.extend([
                "insurance material misrepresentation cases",
                "concealment fraud insurance law",
                "voiding insurance policy misrepresentation"
            ])
        
        # Add general fraud detection queries
        search_queries.extend([
            "insurance fraud detection techniques 2024",
            "red flags insurance claims investigation",
            f"{claim_type} insurance claim fraud patterns"
        ])
        
        # Execute searches
        for query in search_queries[:4]:  # Limit to 4 searches to avoid rate limits
            try:
                results = self.perform_valyu_search(query)
                search_results[query] = results
            except Exception as e:
                search_results[query] = f"Search failed: {str(e)}"
        
        return search_results
    
    def _build_investigation_prompt(self, claim_data: Dict, claim_id: str, conversation_context: str, search_results: Dict) -> str:
        """Build comprehensive investigation prompt with search results"""
        
        # Format search results for the prompt
        search_context = ""
        if search_results:
            search_context = "\n\nEXTERNAL RESEARCH FINDINGS:\n"
            for query, results in search_results.items():
                search_context += f"\nSearch: '{query}'\n"
                search_context += f"Results: {results}\n{'-'*50}"
        else:
            search_context = "\n\nNote: No external research was conducted for this investigation."
        
        prompt = f"""{CORE_INSURANCE_PROTOCOL}

AI Agent Task: SIU Investigator -----

You are a Special Investigations Unit (SIU) Investigator. Your primary directive
is to seek the truth and prevent fraud. You are the company's expert defense
against those who would exploit the system.

Your core value is professional skepticism. You are not cynical, but you are
analytical. You understand that things are not always as they seem. You are an
objective, neutral fact-finder, and your investigations must be thorough, unbiased,
and relentlessly factual.

Your mindset is that of a detective:  
- Red Flags are Your Starting Point: While an adjuster sees a "problem," you see
  a "red flag" as the beginning of an investigation.
- Truth Over Speed: Your priority is accuracy, not efficiency.
- Find the Inconsistency: Find the "delta" — the gap between the claimant's story,
  the physical evidence, and common sense.
- Protect the Honest Customer: Fraudulent claims raise premiums for everyone.

Operational Directives & Observability:  
- Show Your Work: 100% transparency in your reasoning
- Always Justify: Explain why you reach conclusions
- Cite the Evidence: Point to specific, observable facts
- Reference external research when it informs your judgment
- Provide Recommendations with confidence interval (0% to 100%)

Response Style (Slack Conversation):
- Be concise and conversational (2–5 short paragraphs)
- Start with clear summary: "Verdict: LEGITIMATE/SUSPICIOUS/FRAUDULENT (X% confidence)"
- Highlight only the most relevant evidence or red flags
- Mention how online search aided your judgment when relevant
- Keep it tight but justified and professional

{conversation_context if conversation_context != "No previous conversation." else ""}

CLAIM TO INVESTIGATE:
- Claim ID: {claim_id}
- Claimant: {claim_data.get('claimant_name', 'Unknown')}
- Type: {claim_data.get('claim_type', 'Unknown')}
- Amount: ${claim_data.get('claim_amount', 0):,.2f}
- Description: {claim_data.get('description', 'No description provided')}

{search_context}

INVESTIGATION TASK:
Analyze this claim considering:
1. Internal evidence from the claim description
2. External research findings above
3. Industry patterns and red flags
4. Legal and regulatory context

Specifically mention in your analysis:
- How the external research influenced your assessment (if applicable)
- Any patterns or precedents found in the search results
- Whether the search confirmed or contradicted initial suspicions

Provide your final assessment with confidence level.
"""

        return prompt

class ClaimsAdjusterAgent(AdvancedAgent):
    """
    Claims Adjuster
    
    Focus: Policy review, legal risk (bad faith), valuation, file closure
    Constraint: Managing 150+ files, fear of bad faith lawsuit
    Output: Coverage determination and settlement recommendation
    """
    
    def __init__(self, model_id: str = "amazon.nova-micro-v1:0"):
        super().__init__(
            name="ClaimsAdjuster",
            role="Claims Adjuster - Risk Manager",
            model_id=model_id
        )
        self.adjustments = {}
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Handle claims adjustment request"""
        
        claim_data = message.content.get("claim", {})
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   💼 Adjuster reviewing {claim_id}")
        
        # Get conversation context (see SIU findings)
        conversation_context = self._get_conversation_context(message.thread_id)
        
        prompt = f"""{CORE_INSURANCE_PROTOCOL}

AI Agent Task: Claims Adjuster -----

You are a Claims Adjuster. Your job is to be the company's frontline investigator,
negotiator, and first responder to a loss. Your world revolves on closing files
as quickly and efficiently as possible.

Your process:
1. First Contact: Empathetic but firm, immediately start investigation
2. Policy Review: The policy contract is your bible - decisions based on legal wording
3. Investigation & Documentation: "If it's not in the file, it didn't happen"
4. Evaluation: Exact value — not a penny more, not a penny less
5. Settlement & Closure: A good file is a closed file

Key Constraints:
- Managing 150+ files (massive caseload pressure)
- Fear of "bad faith" lawsuit (multi-million dollar risk)
- Suspicion is NOT proof - need ironclad evidence to deny
- Overpaying is "claims leakage"; underpaying is "bad faith"

Your mindset: Fair, not generous. Indemnify based on contract, manage legal risk, close file.

Response Style (Slack):
- Concise (2–5 short paragraphs)
- Start with summary: "Summary: Covered/Denied/Partial — recommend settlement at £X"
- Highlight: coverage point, valuation logic, legal risk
- Keep conversational but policy-based

PREVIOUS CONVERSATION:
{conversation_context}

CURRENT CLAIM:
- Claim ID: {claim_id}
- Claimant: {claim_data.get('claimant_name', 'Unknown')}
- Type: {claim_data.get('claim_type', 'Unknown')}
- Amount Requested: ${claim_data.get('claim_amount', 0):,.2f}
- Description: {claim_data.get('description', 'No description')}

Review this claim considering:
1. Investigation findings above (if any)
2. Bad faith legal exposure
3. Policy coverage
4. Appropriate valuation
5. File closure strategy

Provide your adjustment recommendation in Slack-style format.
"""
        
        response = self.call_model(
            prompt,
            thread_id=message.thread_id,
            include_conversation=False
        )
        
        # Store adjustment
        self.adjustments[claim_id] = response
        
        self.memory.set(
            f"claims_adjustment_{claim_id}",
            response,
            "Claims adjustment with bad faith risk assessment",
            thread_id=message.thread_id
        )
        
        print(f"   ✅ Adjustment complete")
        
        return {
            "agent": self.name,
            "adjustment": response,
            "claim_id": claim_id,
            "status": "completed"
        }


class ClaimsManagerAgent(AdvancedAgent):
    """
    Claims Manager - Final Decision Maker
    
    Focus: Balance legal/bad faith risk vs fraud risk
    Process: Cost-benefit analysis - cost of denying vs cost of paying
    Output: Final binding decision with explicit justification
    """
    
    def __init__(self, model_id: str = "amazon.nova-micro-v1:0"):
        super().__init__(
            name="ClaimsManager",
            role="Claims Manager - Final Authority",
            model_id=model_id
        )
        self.final_decisions = {}
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Make final binding decision"""
        
        claim_data = message.content.get("claim", {})
        claim_id = message.content.get("claim_id", "UNKNOWN")
        
        print(f"   ⚖️  Manager making final decision for {claim_id}")
        
        # Get full conversation (all team input)
        conversation_context = self._get_conversation_context(message.thread_id)
        
        prompt = f"""{CORE_INSURANCE_PROTOCOL}

AI Agent Task: Claims Manager - Final Decision -----

You are the Claim Manager. You are the final decision-maker and the ultimate
risk arbiter. Your primary responsibility is to protect the company's financial
health by balancing two opposing threats: legal/bad faith risk and fraud risk.

You are a pragmatist. Your job is to make a sound business decision.

Your Operational Process:
1. Review the case: Objective summary of claim
2. Facilitate internal debate: Listen to Adjuster (legal risk focus) and Investigator (fraud focus)
3. Ask probing questions:
   - To Adjuster: "What is our bad faith exposure? Defense costs? Nuisance value?"
   - To Investigator: "Is this proof or suspicion? Will it hold up in court? Investigation costs?"
4. Make the Final Decision: Binding call after hearing all sides

Your Core Mandate: The Business Decision

Your entire thought process: What is the total cost of denying this claim (legal fees
+ bad faith judgment risk) versus the total cost of paying it?

You must justify your decision with:
- Clear, pragmatic cost-benefit analysis
- Explicit risk weighing from both team members
- Business reasoning (risk vs cost, not emotion)

Response Style (Slack):
- Concise (2–5 short paragraphs)
- Start with decision: "Decision: APPROVE/DENY/SETTLE at $X — [brief reasoning]"
- Then 2–3 points on:
  - Key legal/bad faith risk
  - Key fraud/investigation finding
  - Cost–benefit logic
- Frame as business decision, not emotion
- Keep conversational but explicitly justified

TEAM ANALYSES:
{conversation_context}

CLAIM DETAILS:
- Claim ID: {claim_id}
- Claimant: {claim_data.get('claimant_name', 'Unknown')}
- Type: {claim_data.get('claim_type', 'Unknown')}
- Amount Requested: ${claim_data.get('claim_amount', 0):,.2f}

Based on all team input above, make your FINAL BINDING DECISION.
Justify it with explicit cost-benefit analysis.
"""
        
        response = self.call_model(
            prompt,
            thread_id=message.thread_id,
            include_conversation=False
        )
        
        # Store decision
        self.final_decisions[claim_id] = response
        
        self.memory.set(
            f"final_decision_{claim_id}",
            response,
            "Final binding decision with cost-benefit justification",
            thread_id=message.thread_id
        )
        
        # Print decision prominently
        print(f"\n{'='*70}")
        print(f"⚖️  CLAIMS MANAGER - FINAL BINDING DECISION")
        print(f"{'='*70}")
        print(f"Claim: {claim_id}")
        print(f"{'='*70}")
        print(response)
        print(f"{'='*70}\n")
        
        return {
            "agent": self.name,
            "decision": response,
            "claim_id": claim_id,
            "status": "completed"
        }