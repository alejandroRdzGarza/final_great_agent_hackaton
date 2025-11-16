# org/message_bus.py
"""
Advanced Message Bus for Multi-Agent Communication

Features:
- Asynchronous message delivery
- Message queues per agent
- Priority handling
- Message routing
- Broadcast capabilities
- Message history tracking
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from queue import PriorityQueue, Queue
import threading
import uuid


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class MessageType(Enum):
    """Types of messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HANDOFF = "handoff"


@dataclass
class Message:
    """Enhanced message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    sender: str = ""
    receiver: str = ""
    thread_id: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_message_id: Optional[str] = None
    requires_response: bool = False
    
    def __lt__(self, other):
        """For priority queue sorting"""
        return self.priority.value < other.priority.value


class MessageBus:
    """
    Central message bus for multi-agent communication.
    
    Handles:
    - Message routing between agents
    - Priority queues per agent
    - Async message delivery
    - Message history
    - Thread tracking
    """
    
    def __init__(self):
        # Agent message queues: agent_name -> PriorityQueue
        self.agent_queues: Dict[str, PriorityQueue] = {}
        
        # Registered agents: agent_name -> agent_instance
        self.agents: Dict[str, Any] = {}
        
        # Message history for tracing
        self.message_history: List[Message] = []
        
        # Thread tracking: thread_id -> list of message_ids
        self.threads: Dict[str, List[str]] = {}
        
        # Callbacks: message_type -> list of callbacks
        self.callbacks: Dict[MessageType, List[Callable]] = {
            mt: [] for mt in MessageType
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        print("🚌 Message Bus initialized")
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent with the message bus"""
        with self.lock:
            if agent_name not in self.agent_queues:
                self.agent_queues[agent_name] = PriorityQueue()
                self.agents[agent_name] = agent_instance
                print(f"   ✅ Registered: {agent_name}")
            else:
                print(f"   ⚠️  Agent {agent_name} already registered")
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent"""
        with self.lock:
            if agent_name in self.agent_queues:
                del self.agent_queues[agent_name]
                del self.agents[agent_name]
                print(f"   ❌ Unregistered: {agent_name}")
    
    def send(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        thread_id: Optional[str] = None,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_response: bool = False,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        Send a message from one agent to another.
        
        Args:
            sender: Sending agent name
            receiver: Receiving agent name
            content: Message content (dict)
            thread_id: Thread/conversation ID
            message_type: Type of message
            priority: Message priority
            requires_response: Whether sender expects a response
            metadata: Additional metadata
        
        Returns:
            Message object that was sent
        """
        
        # Create message
        message = Message(
            type=message_type,
            priority=priority,
            sender=sender,
            receiver=receiver,
            thread_id=thread_id,
            content=content,
            metadata=metadata or {},
            requires_response=requires_response
        )
        
        # Route message
        self._route_message(message)
        
        # Log
        print(f"📨 [{sender}] → [{receiver}]: {message_type.value} (priority: {priority.name})")
        
        return message
    
    def broadcast(
        self,
        sender: str,
        content: Dict[str, Any],
        thread_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude: Optional[List[str]] = None
    ) -> List[Message]:
        """
        Broadcast a message to all registered agents.
        
        Args:
            sender: Sending agent name
            content: Message content
            thread_id: Thread ID
            priority: Message priority
            exclude: List of agent names to exclude
        
        Returns:
            List of messages sent
        """
        
        exclude = exclude or []
        messages = []
        
        with self.lock:
            receivers = [name for name in self.agents.keys() if name not in exclude and name != sender]
        
        for receiver in receivers:
            msg = self.send(
                sender=sender,
                receiver=receiver,
                content=content,
                thread_id=thread_id,
                message_type=MessageType.BROADCAST,
                priority=priority
            )
            messages.append(msg)
        
        print(f"📢 [{sender}] broadcasted to {len(messages)} agents")
        
        return messages
    
    def receive(self, agent_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive next message for an agent (blocking).
        
        Args:
            agent_name: Agent name
            timeout: Timeout in seconds (None = block forever)
        
        Returns:
            Next message or None if timeout
        """
        
        if agent_name not in self.agent_queues:
            print(f"⚠️  Agent {agent_name} not registered")
            return None
        
        try:
            queue = self.agent_queues[agent_name]
            message = queue.get(timeout=timeout) if timeout else queue.get()
            
            print(f"📬 [{agent_name}] received: {message.type.value} from {message.sender}")
            
            return message
        except:
            return None
    
    def has_messages(self, agent_name: str) -> bool:
        """Check if agent has pending messages"""
        if agent_name not in self.agent_queues:
            return False
        return not self.agent_queues[agent_name].empty()
    
    def pending_count(self, agent_name: str) -> int:
        """Get count of pending messages for agent"""
        if agent_name not in self.agent_queues:
            return 0
        return self.agent_queues[agent_name].qsize()
    
    def _route_message(self, message: Message):
        """Route message to appropriate queue"""
        with self.lock:
            # Add to receiver's queue
            if message.receiver in self.agent_queues:
                self.agent_queues[message.receiver].put(message)
            else:
                print(f"⚠️  Receiver {message.receiver} not registered - message dropped")
                return
            
            # Track in history
            self.message_history.append(message)
            
            # Track in thread
            if message.thread_id:
                if message.thread_id not in self.threads:
                    self.threads[message.thread_id] = []
                self.threads[message.thread_id].append(message.id)
            
            # Trigger callbacks
            for callback in self.callbacks.get(message.type, []):
                try:
                    callback(message)
                except Exception as e:
                    print(f"⚠️  Callback error: {e}")
    
    def register_callback(self, message_type: MessageType, callback: Callable):
        """Register a callback for specific message type"""
        self.callbacks[message_type].append(callback)
        print(f"   ✅ Registered callback for {message_type.value}")
    
    def get_thread_messages(self, thread_id: str) -> List[Message]:
        """Get all messages in a thread"""
        if thread_id not in self.threads:
            return []
        
        message_ids = self.threads[thread_id]
        return [msg for msg in self.message_history if msg.id in message_ids]
    
    def get_conversation(self, agent1: str, agent2: str) -> List[Message]:
        """Get conversation between two agents"""
        return [
            msg for msg in self.message_history
            if (msg.sender == agent1 and msg.receiver == agent2) or
               (msg.sender == agent2 and msg.receiver == agent1)
        ]
    
    def get_stats(self) -> Dict:
        """Get message bus statistics"""
        with self.lock:
            return {
                "registered_agents": len(self.agents),
                "total_messages": len(self.message_history),
                "active_threads": len(self.threads),
                "pending_by_agent": {
                    name: queue.qsize()
                    for name, queue in self.agent_queues.items()
                }
            }
    
    def clear_history(self):
        """Clear message history (for testing)"""
        with self.lock:
            self.message_history.clear()
            self.threads.clear()
        print("🗑️  Message history cleared")