# main_organization.py
from org.organization import Organization
from org.tasks import Task
from org.communication import CommunicationHub
from agents.risk_analyst import RiskAnalystAgent
from agents.organizationl_agent import OrganizationalAgent
from agents.planner_agent import PlannerAgent
import os
from pathlib import Path
from dotenv import load_dotenv
import os
import time
from langsmith import uuid7
from dotenv import load_dotenv

import os

# main_organization.py
from pathlib import Path
from dotenv import load_dotenv
import os

# ============================================
# CRITICAL: Load .env FIRST, then set any overrides
# ============================================
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded .env file")
else:
    print("⚠️  No .env file found")

# Override or set if not in .env (make sure these match LANGCHAIN_ not LANGSMITH_)
if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = "your-key-here"  # UPDATE THIS!

os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Must be exactly "true"
if not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "insurance-multi-agent"

# Verify configuration
print("=" * 70)
print("🔍 LangSmith Configuration")
print("=" * 70)

api_key = os.getenv("LANGCHAIN_API_KEY")
tracing = os.getenv("LANGCHAIN_TRACING_V2")
project = os.getenv("LANGCHAIN_PROJECT")

if api_key and len(api_key) > 10 and api_key != "your-key-here":
    print(f"✅ LANGCHAIN_API_KEY: {api_key[:10]}...{api_key[-4:]}")
else:
    print(f"❌ LANGCHAIN_API_KEY: NOT SET - Update main_organization.py line 18!")
    
print(f"✅ LANGCHAIN_TRACING_V2: {tracing}")
print(f"✅ LANGCHAIN_PROJECT: {project}")
print("=" * 70)

# Now import everything else...
from langsmith import uuid7
from org.organization import Organization
from org.tasks import InsuranceClaim, Task
from org.communication import CommunicationHub
from agents.risk_analyst import RiskAnalystAgent
from agents.organizationl_agent import OrganizationalAgent
from agents.planner_agent import PlannerAgent

print("✅ All imports successful!\n")

# ============================================
# CRITICAL: Use LANGCHAIN_* not LANGSMITH_*
# ============================================

# Option 1: Set directly in code (for testing)
os.environ["LANGCHAIN_API_KEY"] = "your-key-here"  # Get from https://smith.langchain.com
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Must be "true" not "1"
os.environ["LANGCHAIN_PROJECT"] = "insurance-multi-agent"  # Optional project name
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # Optional

# Option 2: Load from .env file
# Create a .env file with:
# LANGCHAIN_API_KEY=your-key-here
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=insurance-multi-agent

from dotenv import load_dotenv
load_dotenv()

# Verify configuration
print("=" * 70)
print("🔍 LangSmith Configuration Check")
print("=" * 70)

required_vars = {
    "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
    "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
    "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "default"),
}

all_configured = True
for var_name, var_value in required_vars.items():
    if var_value:
        if var_name == "LANGCHAIN_API_KEY":
            masked = f"{var_value[:10]}...{var_value[-4:]}" if len(var_value) > 14 else "***"
            print(f"✅ {var_name}: {masked}")
        else:
            print(f"✅ {var_name}: {var_value}")
    else:
        print(f"❌ {var_name}: NOT SET")
        all_configured = False

if all_configured:
    print("\n✅ LangSmith is properly configured!")
    print("   All traces will automatically appear at: https://smith.langchain.com")
else:
    print("\n❌ LangSmith configuration incomplete!")
    print("   Set the missing environment variables above.")

print("=" * 70)

# ============================================
# OPTION 1: Set API keys directly (Quick Start)
# ============================================
# Uncomment and set your keys here:
# Recommended: Holistic AI Bedrock
# os.environ["HOLISTIC_AI_TEAM_ID"] = "your-team-id-here"
# os.environ["HOLISTIC_AI_API_TOKEN"] = "your-api-token-here"
# Optional: OpenAI
# os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
# os.environ["LANGSMITH_API_KEY"] = "your-langsmith-key-here"  # Required for LangSmith tracing
# os.environ["LANGSMITH_PROJECT"] = "hackathon-2026"  # Optional
# os.environ["LANGSMITH_TRACING"] = "true"  # Required for LangGraph tracing
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"  # Optional (default)

# ============================================
# OPTION 2: Load from .env file (Recommended)
# ============================================
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("Loaded configuration from .env file")
else:
    print("WARNING: No .env file found - using environment variables or hardcoded keys")

# Import official packages
# Import Holistic AI Bedrock helper function
# Import from core module
try:
    import sys
    # Import from same directory
    from holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model
    print("✅ Holistic AI Bedrock helper function loaded")
except ImportError:
    print("⚠️  Could not import holistic_ai_bedrock")

from langgraph.prebuilt import create_react_agent
# Optional: OpenAI (if not using Bedrock)
# from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Import monitoring tools
import tiktoken
from codecarbon import EmissionsTracker
from langsmith import Client

# Check API keys (Holistic AI Bedrock recommended, OpenAI optional)
# Check Holistic AI Bedrock (recommended)
if os.getenv('HOLISTIC_AI_TEAM_ID') and os.getenv('HOLISTIC_AI_API_TOKEN'):
    print("✅ Holistic AI Bedrock credentials loaded (will use Bedrock)")

# Check OpenAI (optional)
if os.getenv('OPENAI_API_KEY'):
    key_preview = os.getenv('OPENAI_API_KEY')[:10] + "..."
    print(f"OpenAI API key loaded: {key_preview}")
else:
    print("ℹ️  OpenAI API key not found - optional, will use Bedrock if available")

# Check LangSmith API key (required for this tutorial)
if os.getenv('LANGSMITH_API_KEY'):
    ls_key = os.getenv('LANGSMITH_API_KEY')[:10] + "..."
    print(f"LangSmith API key loaded: {ls_key}")
    
    # Set required LangGraph tracing environment variables if not already set
    if not os.getenv('LANGSMITH_TRACING'):
        os.environ['LANGSMITH_TRACING'] = 'true'
        print("  LANGSMITH_TRACING set to 'true' (required for LangGraph tracing)")
    
    if not os.getenv('LANGSMITH_ENDPOINT'):
        os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
        print("  LANGSMITH_ENDPOINT set to 'https://api.smith.langchain.com'")
    
    langsmith_project = os.getenv('LANGSMITH_PROJECT', 'default')
    print(f"  LangSmith project: {langsmith_project}")
    print("  LangSmith tracing will be fully functional!")
else:
    print("ERROR: LangSmith API key not found - tracing will not work!")
    print("  Get a free key at: https://smith.langchain.com")
    print("  This tutorial requires LangSmith to function properly!")

print("\nAll imports successful!")



from pathlib import Path
from dotenv import load_dotenv
import os
from langsmith import uuid7
import time

# ============================================
# Load environment variables
# ============================================
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("Loaded configuration from .env file")
else:
    print("WARNING: No .env file found - using environment variables or hardcoded keys")

# ============================================
# Imports
# ============================================
from org.organization import Organization
from org.tasks import InsuranceClaim, Task
from org.communication import CommunicationHub
from agents.risk_analyst import RiskAnalystAgent
from agents.organizationl_agent import OrganizationalAgent
from agents.planner_agent import PlannerAgent









# main_hackathon.py - Complete Glass Box Solution for Track B
"""
Glass Box Corporation - Transparent Insurance Claims Processing
Track B: Agent Glass Box - Observability, Explainability, Transparency

This system demonstrates:
✅ Complete traceability via LangSmith threads
✅ Multi-agent deliberation with debate system
✅ Automated transparency auditing
✅ Memory update tracking
✅ Valyu AI search integration
✅ Professional report generation
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import uuid7

# Load environment
load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
print("=" * 70)
print("🔍 Glass Box System - Track B Submission")
print("=" * 70)
print(f"✅ LangSmith Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
print("=" * 70)

# Imports
from org.organization import Organization
from org.tasks import InsuranceClaim, Task
from org.communication import CommunicationHub
from org.memory import TrackedMemory
from org.debate import DebateOrchestrator
from agents.organizationl_agent import OrganizationalAgent
from agents.planner_agent import PlannerAgent
from agents.risk_analyst import RiskAnalystAgent
from agents.auditor_agent import AuditorAgent
from utils.report_generator import ReportGenerator
from agents.investigator_agent import InvestigatorAgent











# ============================================
# 1️⃣ Initialize System
# ============================================
print("\n🏗️  Initializing Glass Box System...\n")

hub = CommunicationHub()
org = Organization("GlassBoxCorp")

# Create departments
claims_dept = org.add_department("Claims")
risk_dept = org.add_department("Risk Analysis")
audit_dept = org.add_department("Audit")

# ============================================
# 2️⃣ Create Specialized Agents
# ============================================
print("🤖 Creating specialized agents...\n")

# Use smaller model for efficiency (hackathon requirement)
MODEL = "amazon.nova-micro-v1:0"

claims_agent = OrganizationalAgent("ClaimsValidator", model_id=MODEL)
claims_agent.memory = TrackedMemory("ClaimsValidator")
claims_dept.add_agent(claims_agent)

risk_agent = OrganizationalAgent("RiskAnalyst", model_id=MODEL)
risk_agent.memory = TrackedMemory("RiskAnalyst")
risk_dept.add_agent(risk_agent)

financial_agent = OrganizationalAgent("FinancialAnalyst", model_id=MODEL)
financial_agent.memory = TrackedMemory("FinancialAnalyst")
claims_dept.add_agent(financial_agent)


investigator = InvestigatorAgent("Investigator", model_id=MODEL)
investigator.memory = TrackedMemory("Investigator")
audit_dept.add_agent(investigator)

# Auditor agent for transparency
auditor = AuditorAgent("TransparencyAuditor", model_id=MODEL)
audit_dept.add_agent(auditor)

# Coordinator
all_agents = [claims_agent, risk_agent, financial_agent, investigator]
planner = PlannerAgent(
    name="Coordinator",
    model_id=MODEL,
    communication_hub=hub,
    agents=all_agents
)

print(f"✅ Created {len(all_agents) + 2} agents\n")

# ============================================
# 3️⃣ Create Test Insurance Claim
# ============================================
print("📄 Creating insurance claim...\n")

claim_thread_id = str(uuid7())
print(f"🔗 Thread ID: {claim_thread_id}\n")

claim_data = InsuranceClaim(
    claim_id="HC2025-001",
    claimant_name="Sarah Johnson",
    claim_type="Auto",
    claim_amount=12500.00,
    description=(
        "Multi-vehicle accident on 2025-11-14. "
        "Rear-ended while stopped at traffic light. "
        "Significant damage to rear bumper, trunk, and rear suspension. "
        "Two passengers reported minor whiplash. "
        "Police report filed. Third-party witnesses present."
    ),
    status="Pending"
)
claim_data.thread_id = claim_thread_id




# ============================================
# 5️⃣ Task Planning & Execution
# ============================================
print("=" * 70)
print("📋 PHASE 1: TASK PLANNING")
print("=" * 70 + "\n")

task = Task(
    description=f"Process insurance claim #{claim_data.claim_id}: validate claim, assess risk, calculate payout, make final decision",
    assigned_agent=planner
)
task.claim = claim_data
task.thread_id = claim_thread_id

# Plan and dispatch
planner.plan({
    "description": task.description,
    "claim": claim_data,
    "thread_id": claim_thread_id
})

# ============================================
# 6️⃣ Agent Processing with Rate Limiting
# ============================================
print("\n" + "=" * 70)
print("⚙️  PHASE 2: AGENT PROCESSING")
print("=" * 70 + "\n")

import time

MAX_ROUNDS = 3  # Rate limiting
for round_num in range(MAX_ROUNDS):
    print(f"--- Round {round_num + 1}/{MAX_ROUNDS} ---\n")
    
    for agent in all_agents:
        if hasattr(agent, "process_next_message"):
            agent.process_next_message(hub)
            time.sleep(0.5)  # Rate limiting
    
    print()

# ============================================
# 7️⃣ Multi-Agent Debate
# ============================================
print("=" * 70)
print("🎭 PHASE 3: MULTI-AGENT DELIBERATION")
print("=" * 70 + "\n")

debate_orchestrator = DebateOrchestrator(max_rounds=2)
debate_summary = debate_orchestrator.conduct_debate(
    topic=f"Should claim {claim_data.claim_id} be approved for ${claim_data.claim_amount:,.2f}?",
    agents=[claims_agent, risk_agent, financial_agent],
    thread_id=claim_thread_id
)

print(f"\n✅ Debate complete: {debate_summary['total_rounds']} rounds")
print(f"   Consensus: {'✅ Yes' if debate_summary['consensus_reached'] else '❌ No'}\n")

# ============================================
# 8️⃣ Transparency Auditing
# ============================================
print("=" * 70)
print("🔍 PHASE 4: TRANSPARENCY AUDIT")
print("=" * 70 + "\n")

# Audit each agent's final decision
for agent in all_agents:
    if agent.memory.memory:
        # Create decision summary from memory
        decision_data = {
            "agent_name": agent.name,
            "memory_state": agent.memory.get_summary(),
            "message_count": len([m for m in hub.messages if m.sender == agent])
        }
        
        audit_report = auditor.audit_decision(decision_data, thread_id=claim_thread_id)
        print()

auditor_summary = auditor.generate_audit_summary()
print(f"\n📊 Audit Summary:")
print(f"   Avg Transparency: {auditor_summary['avg_transparency_score']:.2f}/1.0")
print(f"   Avg Shortcut Risk: {auditor_summary['avg_shortcut_risk']:.2f}/1.0")
print(f"   Total Flags: {auditor_summary['total_flags']}\n")

# ============================================
# 9️⃣ Generate Final Report
# ============================================
print("=" * 70)
print("📄 PHASE 5: REPORT GENERATION")
print("=" * 70 + "\n")

report_gen = ReportGenerator(
    project_name="Glass Box Insurance Claims Processing",
    team_name="Your Team Name"  # UPDATE THIS
)

final_decision = debate_summary.get('final_decision', "Decision pending further review")

report = report_gen.generate_report(
    claim_data=claim_data.__dict__,
    agents_used=[a.name for a in all_agents + [auditor, planner]],
    auditor_summary=auditor_summary,
    debate_summary=debate_summary,
    trace_data={
        "project": os.getenv('LANGCHAIN_PROJECT', 'default'),
        "thread_id": claim_thread_id,
        "events": [
            {"agent": m.sender.name, "action": m.content[:50], "timestamp": str(m.timestamp)}
            for m in hub.messages[:10]
        ]
    },
    final_decision=final_decision
)

# Save report
report_gen.save_report(report, "glass_box_final_report.md")

print(f"✅ Report generated!\n")

# ============================================
# 🔟 Summary & Next Steps
# ============================================
print("=" * 70)
print("✅ GLASS BOX PROCESSING COMPLETE")
print("=" * 70)
print(f"\n📊 System Summary:")
print(f"   Claim ID: {claim_data.claim_id}")
print(f"   Thread ID: {claim_thread_id}")
print(f"   Agents Used: {len(all_agents) + 2}")
print(f"   Messages Exchanged: {len(hub.messages)}")
print(f"   Debate Rounds: {debate_summary['total_rounds']}")
print(f"   Transparency Score: {auditor_summary['avg_transparency_score']:.2f}/1.0")
print(f"   Memory Updates: {sum(len(a.memory.history) for a in all_agents if hasattr(a, 'memory'))}")
print(f"\n🔗 View in LangSmith:")
print(f"   https://smith.langchain.com")
print(f"   Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
print(f"   Thread ID: {claim_thread_id}")
print(f"\n📄 Report: glass_box_final_report.md")
print("=" * 70)