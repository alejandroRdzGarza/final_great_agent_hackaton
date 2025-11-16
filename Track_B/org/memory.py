# org/memory.py
from org.schemas import MemoryUpdate
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class TrackedMemory:
    """
    Memory system that tracks all updates for LangSmith visibility.
    
    Every memory change is logged and can be traced.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.memory: Dict[str, Any] = {}
        self.history: List[MemoryUpdate] = []
    
    def set(self, key: str, value: Any, reasoning: str, thread_id: str = None):
        """Set or update a memory value with tracking"""
        
        old_value = self.memory.get(key)
        update_type = "modify" if key in self.memory else "add"
        
        # Create memory update record
        update = MemoryUpdate(
            agent_name=self.agent_name,
            update_type=update_type,
            key=key,
            old_value=old_value,
            new_value=value,
            reasoning=reasoning
        )
        
        # Update memory
        self.memory[key] = value
        self.history.append(update)
        
        # Log for LangSmith visibility
        print(f"💾 [{self.agent_name}] Memory {update_type}: {key}")
        print(f"   Reasoning: {reasoning}")
        if thread_id:
            print(f"   Thread: {thread_id}")
        
        return update
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a memory value"""
        return self.memory.get(key, default)
    
    def delete(self, key: str, reasoning: str):
        """Delete a memory value with tracking"""
        
        if key not in self.memory:
            return None
        
        old_value = self.memory[key]
        
        update = MemoryUpdate(
            agent_name=self.agent_name,
            update_type="delete",
            key=key,
            old_value=old_value,
            new_value=None,
            reasoning=reasoning
        )
        
        del self.memory[key]
        self.history.append(update)
        
        print(f"💾 [{self.agent_name}] Memory deleted: {key}")
        print(f"   Reasoning: {reasoning}")
        
        return update
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get memory update history"""
        history = [h.dict() for h in self.history]
        if limit:
            return history[-limit:]
        return history
    
    def get_summary(self) -> Dict:
        """Get memory summary"""
        return {
            "agent": self.agent_name,
            "total_keys": len(self.memory),
            "total_updates": len(self.history),
            "current_state": self.memory.copy(),
            "recent_updates": self.get_history(limit=5)
        }
    def append(self, item: dict):
        """
        Append generic items to memory history.
        Used for logging messages, interactions, or events.
        """
        # Store it in history for LangSmith visibility
        update = MemoryUpdate(
            agent_name=self.agent_name,
            update_type="log",
            key=item.get("key", "event"),
            old_value=None,
            new_value=item,
            reasoning=item.get("reasoning", "Logged event")
        )

        self.history.append(update)

        print(f"📝 [{self.agent_name}] Logged event:")
        print(f"    {item}")

        return update
