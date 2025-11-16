# org/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ReasoningStep(BaseModel):
    """Single step in agent reasoning"""
    step_number: int
    action: str = Field(description="What the agent did")
    reasoning: str = Field(description="Why the agent did it")
    tool_used: Optional[str] = Field(None, description="Tool called, if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentDecision(BaseModel):
    """Structured output from any agent"""
    agent_name: str
    task: str
    steps: List[ReasoningStep] = Field(description="Reasoning trace")
    reasoning: str = Field(description="Overall reasoning")
    decision: str = Field(description="Final decision/output")
    citations: List[str] = Field(default_factory=list, description="Sources used")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    handoff_message: Optional[str] = Field(None, description="Message for next agent")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TransparencyReport(BaseModel):
    """Auditor's transparency assessment"""
    agent_name: str
    transparency_score: float = Field(ge=0, le=1)
    risk_of_shortcut_score: float = Field(ge=0, le=1)
    reasoning_timeline: List[ReasoningStep]
    flags: List[str] = Field(default_factory=list, description="Issues found")
    recommendations: List[str] = Field(default_factory=list)

class MemoryUpdate(BaseModel):
    """Track memory changes"""
    agent_name: str
    update_type: str = Field(description="add/modify/delete")
    key: str
    old_value: Optional[Any] = None
    new_value: Any
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)