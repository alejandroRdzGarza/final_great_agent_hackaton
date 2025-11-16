import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from utils.orchestrator import MultiAgentOrchestrator
from org.tasks import InsuranceClaim

# Load environment
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("=" * 70)
print("🔷 GLASS BOX MULTI-AGENT SYSTEM")
print("=" * 70)
print(f"   Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70 + "\n")

# Import your agents
from agents.investigator_agent import InvestigatorAgent
from agents.risk_analyst import RiskAnalystAgent
from agents.finance_agent import FinancialAnalystAgent
from agents.auditor_agent import AuditorAgent
from agents.advanced_agent import AdvancedAgent
from agents.insurance_agents import SIUInvestigatorAgent, ClaimsAdjusterAgent, ClaimsManagerAgent

def main():
    """Main execution with insurance-specific agents and agent thoughts display"""
    
    print("🏗️  Initializing Insurance Claims System...\n")
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Create insurance-specific agents
    print("🤖 Creating insurance agents...\n")
    
    siu_investigator = SIUInvestigatorAgent()
    claims_adjuster = ClaimsAdjusterAgent()
    auditor = AuditorAgent()
    claims_manager = ClaimsManagerAgent()
    
    # Register agents
    orchestrator.register_agent(siu_investigator)
    orchestrator.register_agent(claims_adjuster)
    orchestrator.register_agent(auditor)
    orchestrator.register_agent(claims_manager)
    
    print(f"\n✅ System ready with {len(orchestrator.agents)} agents\n")
    
    # Create a realistic insurance claim
    print("📄 Creating insurance claim...\n")
    
    claim = InsuranceClaim(
        claim_id="FRAUD-2025-001",
        claimant_name="John Smith",
        claim_type="Auto Theft",
        claim_amount=25000.00,
        description=(
            "Customer reports that their car was stolen from their driveway at 3:00 AM. "
            "However: "
            "- GPS data shows the vehicle driving continuously 200 km away during that time. "
            "- Traffic camera footage shows the customer themselves driving the car an hour later. "
            "- The customer recently increased the insurance coverage significantly two weeks ago."
        ),
        status="Pending"
    )
    
    print(f"   Claim ID: {claim.claim_id}")
    print(f"   Type: {claim.claim_type}")
    print(f"   Amount: ${claim.claim_amount:,.2f}")
    print(f"   Red Flags: GPS mismatch, camera evidence, recent coverage increase\n")
    
    # Process the claim through the orchestrator workflow
    workflow = orchestrator.process_claim(
        claim_id=claim.claim_id,
        claim_data=claim.__dict__
    )
    
    # Display each agent’s thoughts (the debate)
    print("\n💭 AGENT THOUGHTS / MINI-DEBATE:")
    agent_keys = {
        "SIU_Investigator": f"siu_investigation_{claim.claim_id}",
        "ClaimsAdjuster": f"claims_adjustment_{claim.claim_id}",
        "TransparencyAuditor": f"transparency_audit_{claim.claim_id}"
    }

    for agent_name, memory_key in agent_keys.items():
        agent = orchestrator.agents[agent_name]
        thought = agent.memory.get(memory_key, "No reasoning recorded")
        print("\n" + "-"*50)
        print(f"[{agent_name}] Thought:")
        print(thought)
    print("-"*50)

    
    # Display final decision from Claims Manager
    print("\n⚖️ FINAL DECISION:")
    manager = orchestrator.agents["ClaimsManager"]
    final_decision = manager.final_decisions.get(claim.claim_id, "No decision recorded")
    print(final_decision)
    
    print("\n✅ Claim processing complete")
    
    return workflow, orchestrator

if __name__ == "__main__":
    orchestrator = main()
    print("\n🎉 System demonstration complete!\n")
