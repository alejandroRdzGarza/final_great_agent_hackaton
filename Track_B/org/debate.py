# org/debate.py
from typing import List, Dict, Any
from agents.base_agent import BaseAgent
import json

class DebateRound:
    """Single round of debate between agents"""
    
    def __init__(self, topic: str, round_number: int):
        self.topic = topic
        self.round_number = round_number
        self.positions = []  # List of agent positions
        self.consensus_reached = False
        self.final_decision = None
    
    def add_position(self, agent_name: str, position: str, reasoning: str):
        """Add an agent's position"""
        self.positions.append({
            "agent": agent_name,
            "position": position,
            "reasoning": reasoning,
            "round": self.round_number
        })


class DebateOrchestrator:
    """
    Orchestrates multi-agent debates for complex decisions.
    
    Process:
    1. Present topic to all agents
    2. Each agent states position
    3. Agents challenge each other
    4. Repeat until consensus or max rounds
    5. Final synthesis
    """
    
    def __init__(self, max_rounds: int = 3, consensus_threshold: float = 0.7):
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.debate_history = []
    
    def conduct_debate(
        self, 
        topic: str, 
        agents: List[BaseAgent],
        thread_id: str = None
    ) -> Dict[str, Any]:
        """
        Conduct a multi-round debate between agents.
        
        Returns:
            Debate summary with positions, reasoning, and final decision
        """
        
        print(f"\n🎭 Starting debate on: {topic}")
        print(f"   Participants: {[a.name for a in agents]}")
        print(f"   Max rounds: {self.max_rounds}")
        
        rounds = []
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            
            debate_round = DebateRound(topic, round_num)
            
            # Each agent states their position
            for agent in agents:
                # Build context from previous rounds
                context = self._build_context(rounds, agent.name)
                
                prompt = f"""Topic: {topic}

{context}

As {agent.name}, state your position on this topic. Consider:
1. Your role and expertise
2. Previous positions from other agents (if any)
3. Risks and benefits
4. Evidence and reasoning

Respond in JSON format:
{{
  "position": "agree/disagree/neutral",
  "reasoning": "Your detailed reasoning",
  "key_points": ["point1", "point2", ...],
  "concerns": ["concern1", "concern2", ...]
}}
"""
                
                response = agent.call_model(prompt, thread_id=thread_id)
                
                try:
                    position_data = json.loads(response)
                    debate_round.add_position(
                        agent.name,
                        position_data.get("position", "neutral"),
                        position_data.get("reasoning", "No reasoning provided")
                    )
                    
                    print(f"   {agent.name}: {position_data.get('position')}")
                    
                except json.JSONDecodeError:
                    # Fallback for non-JSON response
                    debate_round.add_position(
                        agent.name,
                        "unclear",
                        response[:200]
                    )
            
            rounds.append(debate_round)
            
            # Check for consensus
            if self._check_consensus(debate_round):
                print(f"   ✅ Consensus reached!")
                debate_round.consensus_reached = True
                break
        
        # Synthesize final decision
        final_decision = self._synthesize_decision(rounds, agents[0], thread_id)
        
        # Store debate
        debate_summary = {
            "topic": topic,
            "participants": [a.name for a in agents],
            "rounds": [self._round_to_dict(r) for r in rounds],
            "consensus_reached": rounds[-1].consensus_reached if rounds else False,
            "final_decision": final_decision,
            "total_rounds": len(rounds)
        }
        
        self.debate_history.append(debate_summary)
        
        return debate_summary
    
    def _build_context(self, rounds: List[DebateRound], current_agent: str) -> str:
        """Build context from previous rounds"""
        if not rounds:
            return "This is the first round. No previous positions."
        
        context = "Previous positions:\n"
        for round in rounds:
            for pos in round.positions:
                if pos["agent"] != current_agent:
                    context += f"- {pos['agent']}: {pos['position']} - {pos['reasoning'][:100]}...\n"
        
        return context
    
    def _check_consensus(self, debate_round: DebateRound) -> bool:
        """Check if consensus has been reached"""
        positions = [p["position"] for p in debate_round.positions]
        
        if not positions:
            return False
        
        # Count most common position
        from collections import Counter
        position_counts = Counter(positions)
        most_common_count = position_counts.most_common(1)[0][1]
        
        consensus_ratio = most_common_count / len(positions)
        
        return consensus_ratio >= self.consensus_threshold
    
    def _synthesize_decision(
        self, 
        rounds: List[DebateRound], 
        synthesizer: BaseAgent,
        thread_id: str
    ) -> str:
        """Synthesize final decision from all rounds"""
        
        print(f"\n🎯 Synthesizing final decision...")
        
        # Build summary of all positions
        summary = "Debate Summary:\n\n"
        for round in rounds:
            summary += f"Round {round.round_number}:\n"
            for pos in round.positions:
                summary += f"- {pos['agent']}: {pos['position']} - {pos['reasoning'][:100]}...\n"
            summary += "\n"
        
        prompt = f"""{summary}

Based on this multi-round debate, synthesize a final decision that:
1. Incorporates key insights from all participants
2. Addresses main concerns raised
3. Provides clear reasoning
4. Suggests next steps

Provide a balanced, well-reasoned final decision.
"""
        
        final_decision = synthesizer.call_model(prompt, thread_id=thread_id)
        
        return final_decision
    
    def _round_to_dict(self, round: DebateRound) -> Dict:
        """Convert DebateRound to dict"""
        return {
            "round_number": round.round_number,
            "topic": round.topic,
            "positions": round.positions,
            "consensus_reached": round.consensus_reached
        }
    
    def get_debate_summary(self) -> Dict:
        """Get summary of all debates"""
        return {
            "total_debates": len(self.debate_history),
            "debates": self.debate_history
        }