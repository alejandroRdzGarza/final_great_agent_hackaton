# org/orchestrator.py
"""
Complete Multi-Agent Orchestrator

Features:
- Thread-per-claim management
- Agent coordination
- Message routing
- Workflow execution
- Status tracking
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import time
from langsmith import uuid7

from utils.message_bus import MessageBus, MessageType, MessagePriority
from agents.advanced_agent import AdvancedAgent
from org.schemas import AgentDecision


class ClaimWorkflow:
    """Workflow for processing a single claim"""
    
    def __init__(self, claim_id: str, claim_data: Dict, thread_id: str):
        self.claim_id = claim_id
        self.claim_data = claim_data
        self.thread_id = thread_id
        self.status = "initialized"
        self.stages: List[Dict] = []
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self.results: Dict[str, Any] = {}
    
    def add_stage(self, stage_name: str, agent_name: str, status: str = "pending"):
        """Add a workflow stage"""
        self.stages.append({
            "name": stage_name,
            "agent": agent_name,
            "status": status,
            "started_at": None,
            "completed_at": None
        })
    
    def start_stage(self, stage_name: str):
        """Mark stage as started"""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["status"] = "in_progress"
                stage["started_at"] = datetime.now(timezone.utc)
                break
    
    def complete_stage(self, stage_name: str, result: Any = None):
        """Mark stage as completed"""
        for stage in self.stages:
            if stage["name"] == stage_name:
                stage["status"] = "completed"
                stage["completed_at"] = datetime.now(timezone.utc)
                if result:
                    self.results[stage_name] = result
                break
    
    def get_status(self) -> Dict:
        """Get workflow status"""
        total_stages = len(self.stages)
        completed_stages = sum(1 for s in self.stages if s["status"] == "completed")
        
        return {
            "claim_id": self.claim_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "progress": f"{completed_stages}/{total_stages}",
            "stages": self.stages,
            "started_at": str(self.started_at),
            "completed_at": str(self.completed_at) if self.completed_at else None
        }


class MultiAgentOrchestrator:
    """Orchestrator with message bus registration"""
    
    def __init__(self):
        # Coordinator agent name (must be set before registering)
        self.coordinator_name = "Orchestrator"
        
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Register orchestrator itself to receive responses
        self.message_bus.register_agent(self.coordinator_name, self)
        
        # Registered agents
        self.agents: Dict[str, AdvancedAgent] = {}
        
        # Active workflows
        self.workflows: Dict[str, ClaimWorkflow] = {}
        
        print("🎯 Multi-Agent Orchestrator initialized")
    
    def register_agent(self, agent: AdvancedAgent):
        """Register an agent with the orchestrator"""
        
        # Connect agent to message bus
        agent.connect_to_bus(self.message_bus)
        
        # Store reference
        self.agents[agent.name] = agent
        
        print(f"   ✅ Registered: {agent.name} ({agent.role})")
    
    def process_claim(
        self,
        claim_id: str,
        claim_data: Dict,
        workflow_steps: Optional[List[Dict]] = None
    ) -> ClaimWorkflow:
        """
        Process an insurance claim through multi-agent workflow.
        
        Args:
            claim_id: Unique claim identifier
            claim_data: Claim information
            workflow_steps: Custom workflow steps (or use default)
        
        Returns:
            ClaimWorkflow object for tracking
        """
        
        # Create thread for this claim
        thread_id = str(uuid7())
        
        print("\n" + "=" * 70)
        print(f"🚀 Starting Claim Processing")
        print("=" * 70)
        print(f"   Claim ID: {claim_id}")
        print(f"   Thread ID: {thread_id}")
        print(f"   Amount: ${claim_data.get('claim_amount', 0):,.2f}")
        print("=" * 70 + "\n")
        
        # Create workflow
        workflow = ClaimWorkflow(claim_id, claim_data, thread_id)
        self.workflows[claim_id] = workflow
        
        # Define workflow steps (or use provided)
        if not workflow_steps:
            workflow_steps = self._default_workflow()
        
        # Initialize stages
        for step in workflow_steps:
            workflow.add_stage(step["name"], step["agent"])
        
        workflow.status = "in_progress"
        
        # Execute workflow
        for step in workflow_steps:
            self._execute_step(workflow, step)
        
        # Mark workflow as completed
        workflow.status = "completed"
        workflow.completed_at = datetime.now(timezone.utc)
        
        print("\n" + "=" * 70)
        print("✅ Claim Processing Complete")
        print("=" * 70)
        print(f"   Claim ID: {claim_id}")
        print(f"   Duration: {(workflow.completed_at - workflow.started_at).total_seconds():.2f}s")
        print(f"   Stages: {len(workflow.stages)}")
        print("=" * 70 + "\n")
        
        return workflow
    
    # org/orchestrator.py - Update _default_workflow()

    # org/orchestrator.py - Update the _default_workflow method

    def _default_workflow(self) -> List[Dict]:
        """Insurance-specific workflow matching industry process"""
        return [
            {
                "name": "siu_investigation",
                "agent": "SIU_Investigator",
                "description": "Fraud investigation with confidence assessment"
            },
            {
                "name": "claims_adjustment",
                "agent": "ClaimsAdjuster",
                "description": "Policy review, legal risk, and valuation"
            },
            {
                "name": "transparency_audit",
                "agent": "TransparencyAuditor",
                "description": "Audit reasoning and decision transparency"
            },
            {
                "name": "final_decision",
                "agent": "ClaimsManager",
                "description": "Final binding decision with cost-benefit analysis"
            }
        ]
    
    def _execute_step(self, workflow: ClaimWorkflow, step: Dict):
        """Execute a single workflow step"""
        
        stage_name = step["name"]
        agent_name = step["agent"]
        
        print(f"\n📍 Stage: {stage_name}")
        print(f"   Agent: {agent_name}")
        print("-" * 70)
        
        # Check if agent exists
        if agent_name not in self.agents:
            print(f"   ⚠️  Agent {agent_name} not registered - skipping")
            workflow.complete_stage(stage_name, {"error": "Agent not found"})
            return
        
        agent = self.agents[agent_name]
        
        # Mark stage as started
        workflow.start_stage(stage_name)
        
        # Send task to agent
        message = self.message_bus.send(
            sender=self.coordinator_name,
            receiver=agent_name,
            content={
                "type": stage_name,
                "claim": workflow.claim_data,
                "claim_id": workflow.claim_id,
                "stage": stage_name
            },
            thread_id=workflow.thread_id,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            requires_response=True
        )
        
        # Process agent's messages
        time.sleep(0.5)  # Give agent time to process
        agent.process_messages(max_messages=5, timeout=0.5)
        
        # Wait for response
        response = self._wait_for_response(agent_name, workflow.thread_id, timeout=5.0)
        
        # Mark stage as completed
        workflow.complete_stage(stage_name, response)
        
        print(f"   ✅ {stage_name} completed")
        print("-" * 70)
    
    # org/orchestrator.py - Update the _wait_for_response method

    def _wait_for_response(
        self,
        agent_name: str,
        thread_id: str,
        timeout: float = 5.0
    ) -> Optional[Dict]:
        """Wait for agent response"""
        
        # Give more time for response to arrive
        time.sleep(0.2)
        
        # Check message history for response
        messages = self.message_bus.get_thread_messages(thread_id)
        
        # Look for most recent response from this agent
        for msg in reversed(messages):
            if msg.sender == agent_name and msg.type == MessageType.RESPONSE:
                print(f"   📬 Got response from {agent_name}")
                return msg.content
        
        # If no response in thread, check if agent sent any messages
        print(f"   ⚠️  No response found from {agent_name} in thread")
        return {"status": "no_response", "agent": agent_name}
        
    def get_workflow_status(self, claim_id: str) -> Optional[Dict]:
        """Get status of a workflow"""
        
        if claim_id not in self.workflows:
            return None
        
        workflow = self.workflows[claim_id]
        return workflow.get_status()
    
    def get_all_workflows(self) -> List[Dict]:
        """Get status of all workflows"""
        return [w.get_status() for w in self.workflows.values()]
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        
        total_workflows = len(self.workflows)
        completed = sum(1 for w in self.workflows.values() if w.status == "completed")
        in_progress = sum(1 for w in self.workflows.values() if w.status == "in_progress")
        
        return {
            "registered_agents": len(self.agents),
            "total_workflows": total_workflows,
            "completed_workflows": completed,
            "in_progress_workflows": in_progress,
            "message_bus_stats": self.message_bus.get_stats(),
            "agent_stats": {
                name: agent.get_stats()
                for name, agent in self.agents.items()
            }
        }