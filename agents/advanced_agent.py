# agents/advanced_agent.py
"""
Advanced Agent Base Class with Message Bus Integration

Features:
- Async message processing
- Message bus integration
- Tool calling support
- Structured outputs
- Memory tracking
- LangSmith tracing
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import os
from datetime import datetime, timezone
from langchain_core.runnables import RunnableConfig
from holistic_ai_bedrock import get_chat_model

from utils.message_bus import MessageBus, Message, MessageType, MessagePriority
from org.schemas import AgentDecision, ReasoningStep, MemoryUpdate
from org.memory import TrackedMemory

# agents/advanced_agent.py - Add conversation context

class AdvancedAgent(ABC):
    """Advanced agent with conversation memory"""
    
    def __init__(self, name: str, role: str, model_id: str = "amazon.nova-micro-v1:0", temperature: float = 0.0, tools: Optional[List] = None):
        self.name = name
        self.role = role
        self.model_id = model_id
        self.temperature = temperature
        self.tools = tools or []
        
        # Initialize model
        self.model = get_chat_model(model_id, temperature=temperature)
        
        # Initialize memory
        self.memory = TrackedMemory(name)
        
        # Conversation history per thread
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        # Message bus (set by orchestrator)
        self.message_bus: Optional[MessageBus] = None
        
        # Processing state
        self.is_processing = False
        self.current_thread_id: Optional[str] = None
        
        # Decision history
        self.decisions: List[AgentDecision] = []
        
        print(f"🤖 {name} ({role}) initialized")
    
    def connect_to_bus(self, message_bus: MessageBus):
        """Connect agent to message bus"""
        self.message_bus = message_bus
        message_bus.register_agent(self.name, self)
        print(f"   ✅ {self.name} connected to message bus")
    
    def _add_to_conversation(self, thread_id: str, role: str, content: str):
        """Add message to conversation history"""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        
        self.conversation_history[thread_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    # agents/advanced_agent.py - Update _get_conversation_context

    def _get_conversation_context(self, thread_id: str, format_for_decision: bool = False) -> str:
        """Get full conversation context for this thread"""
        if thread_id not in self.conversation_history:
            return "No previous conversation."
        
        history = self.conversation_history[thread_id]
        
        if format_for_decision:
            # Format specifically for decision maker to read analyses
            context = "=== PREVIOUS TEAM ANALYSES ===\n\n"
            
            for i, msg in enumerate(history, 1):
                role = msg['role']
                content = msg['content']
                
                # If content is a dict, extract the key findings
                if isinstance(content, dict):
                    context += f"--- {role} Analysis ---\n"
                    
                    # Extract investigation
                    if 'investigation' in content:
                        context += f"INVESTIGATION FINDINGS:\n{content['investigation']}\n\n"
                    
                    # Extract risk analysis
                    elif 'analysis' in content and 'risk' in role.lower():
                        context += f"RISK ASSESSMENT:\n{content['analysis']}\n\n"
                    
                    # Extract financial analysis
                    elif 'analysis' in content and 'financial' in role.lower():
                        context += f"FINANCIAL ANALYSIS:\n{content['analysis']}\n\n"
                    
                    # Extract audit
                    elif 'audit' in content:
                        context += f"TRANSPARENCY AUDIT:\n{content['audit']}\n\n"
                    
                    # Generic extraction
                    else:
                        for key, value in content.items():
                            if key not in ['agent', 'status', 'claim_id']:
                                context += f"{key.upper()}: {value}\n"
                        context += "\n"
                else:
                    # String content
                    context += f"[{role}]: {str(content)[:500]}\n\n"
            
            return context
        else:
            # Original format for other agents
            context = "Previous conversation:\n"
            for i, msg in enumerate(history[-10:], 1):
                context += f"{i}. [{msg['role']}]: {str(msg['content'])[:200]}...\n"
            
            return context
    
    def _handle_message(self, message: Message):
        """Handle a received message with conversation tracking"""
        
        print(f"\n⚙️  [{self.name}] Processing message from {message.sender}")
        
        self.current_thread_id = message.thread_id
        
        # Add incoming message to conversation
        if message.thread_id:
            self._add_to_conversation(
                message.thread_id,
                f"{message.sender}",
                str(message.content)
            )
        
        # Route by message type
        if message.type == MessageType.REQUEST:
            response = self.handle_request(message)
            
            # Add our response to conversation
            if message.thread_id and response:
                self._add_to_conversation(
                    message.thread_id,
                    self.name,
                    str(response)
                )
            
            # Send response if required
            if message.requires_response and response:
                self.send_message(
                    receiver=message.sender,
                    content=response,
                    thread_id=message.thread_id,
                    message_type=MessageType.RESPONSE
                )
        
        elif message.type == MessageType.HANDOFF:
            self.handle_handoff(message)
        
        elif message.type == MessageType.BROADCAST:
            self.handle_broadcast(message)
        
        elif message.type == MessageType.RESPONSE:
            self.handle_response(message)
    
    def send_message(
        self,
        receiver: str,
        content: Dict[str, Any],
        thread_id: Optional[str] = None,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL
    ):
        """Send a message via the message bus"""
        if not self.message_bus:
            raise ValueError(f"{self.name} not connected to message bus")
        
        return self.message_bus.send(
            sender=self.name,
            receiver=receiver,
            content=content,
            thread_id=thread_id or self.current_thread_id,
            message_type=message_type,
            priority=priority
        )
    
    def broadcast(
        self,
        content: Dict[str, Any],
        thread_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ):
        """Broadcast message to all agents"""
        if not self.message_bus:
            raise ValueError(f"{self.name} not connected to message bus")
        
        return self.message_bus.broadcast(
            sender=self.name,
            content=content,
            thread_id=thread_id or self.current_thread_id,
            priority=priority,
            exclude=[self.name]
        )
    
    def process_messages(self, max_messages: int = 10, timeout: float = 0.1):
        """
        Process pending messages from queue.
        
        Args:
            max_messages: Max number of messages to process
            timeout: Timeout for each receive call
        """
        if not self.message_bus:
            return
        
        processed = 0
        while processed < max_messages:
            message = self.message_bus.receive(self.name, timeout=timeout)
            
            if message is None:
                break
            
            self._handle_message(message)
            processed += 1
        
        if processed > 0:
            print(f"   ✅ {self.name} processed {processed} messages")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        pending_messages = 0
        if self.message_bus and hasattr(self.message_bus, 'get_queue_size'):
            pending_messages = self.message_bus.get_queue_size(self.name)
        elif self.message_bus and hasattr(self.message_bus, 'agent_queues'):
            queue = self.message_bus.agent_queues.get(self.name)
            pending_messages = queue.qsize() if queue else 0
        
        memory_items = 0
        if hasattr(self.memory, 'memory') and isinstance(self.memory.memory, dict):
            memory_items = len(self.memory.memory)
        elif hasattr(self.memory, 'get_all'):
            try:
                memory_items = len(self.memory.get_all())
            except:
                pass
        
        return {
            "decisions_made": len(self.decisions),
            "memory_items": memory_items,
            "pending_messages": pending_messages,
            "conversation_threads": len(self.conversation_history)
        }
    
    def handle_request(self, message: Message) -> Optional[Dict]:
        """Handle a request message - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement handle_request")
    
    def handle_handoff(self, message: Message):
        """Handle a handoff message - to be implemented by subclasses"""
        pass
    
    def handle_broadcast(self, message: Message):
        """Handle a broadcast message - to be implemented by subclasses"""
        pass
    
    def handle_response(self, message: Message):
        """Handle a response message - to be implemented by subclasses"""
        pass
    
    def call_model(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        include_conversation: bool = True,
        tools: Optional[List] = None,
        structured_output: bool = False
    ) -> str:
        """
        Call the LLM with conversation context and LangSmith tracing.
        
        Args:
            prompt: Input prompt
            thread_id: Thread ID for tracing
            include_conversation: Whether to include conversation history
            tools: Tools to make available
            structured_output: Whether to enforce structured output
        
        Returns:
            Model response
        """
        
        thread_id = thread_id or self.current_thread_id
        
        # Build full prompt with conversation context
        if include_conversation and thread_id:
            conversation_context = self._get_conversation_context(thread_id)
            full_prompt = f"""{conversation_context}

Current task:
{prompt}

Respond considering the full conversation history above."""
        else:
            full_prompt = prompt
        
        print(f"   🤖 {self.name} calling model (with context: {include_conversation})...")
        
        # Configure tracing
        config = RunnableConfig(
            metadata={
                "thread_id": thread_id,
                "agent_name": self.name,
                "agent_role": self.role,
                "conversation_turns": len(self.conversation_history.get(thread_id or "", []))
            },
            tags=[self.name, self.role, "agent_call"]
        )
        
        # Call model
        result = self.model.invoke(full_prompt, config=config)
        response = result.content if hasattr(result, "content") else str(result)
        
        print(f"   ✅ Response: {len(response)} chars")
        
        return response