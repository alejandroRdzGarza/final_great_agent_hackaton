# main_organization.py
from org.organization import Organization
from org.tasks import Task
from org.communication import CommunicationHub
from agents.risk_analyst import RiskAnalystAgent
from agents.base_agent_comms import OrganizationalAgent

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

# Communication example
claims_org_agent.send_message(risk_agent, "Please assess risk for claim #123", hub, task_id=task1.id)

# Access messages
for msg in hub.messages:
    print(f"{msg.timestamp} | {msg.sender.name} -> {msg.receiver.name}: {msg.content}")
