# main_organization.py
from org.organization import Organization
from org.tasks import Task
from org.communication import CommunicationHub
from agents.risk_analyst import RiskAnalystAgent
from agents.base_agent_comms import OrganizationalAgent
import os
from pathlib import Path
from dotenv import load_dotenv

import os
import time
import uuid
from dotenv import load_dotenv

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


# Initialize communication
hub = CommunicationHub()

# Create organization
org = Organization("InsuranceCorp")
claims_dept = org.add_department("Claims")
risk_dept = org.add_department("Risk Analysis")

# Add agents
claims_agent = RiskAnalystAgent()
claims_org_agent = OrganizationalAgent("ClaimsOrg", model_id="amazon.titan-1")
claims_dept.add_agent(claims_org_agent)

risk_agent = OrganizationalAgent("RiskAnalyst", model_id="amazon.titan-1")
risk_dept.add_agent(risk_agent)

# Create task
task1 = Task("Validate insurance claim #123", assigned_agent=claims_org_agent)

# 1️⃣ Send initial message
claims_org_agent.send_message(
    risk_agent, 
    "Please assess risk for claim #123", 
    hub, 
    task_id=task1.id
)

# 2️⃣ Reactive loop: process inboxes
# Simple example: each agent processes one message per iteration
for _ in range(2):  # two iterations, can increase if you want more conversation
    for agent in [claims_org_agent, risk_agent]:
        if hasattr(agent, "process_next_message"):
            agent.process_next_message(hub)

# 3️⃣ Print all messages after processing
for msg in hub.messages:
    print(f"{msg.timestamp} | {msg.sender.name} -> {msg.receiver.name}: {msg.content}")
