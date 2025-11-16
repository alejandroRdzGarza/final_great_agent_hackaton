import os
from dotenv import load_dotenv
from utils.orchestrator import MultiAgentOrchestrator
from org.tasks import InsuranceClaim
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
import time
import re
import uuid
from agents.auditor_agent import AuditorAgent
from agents.insurance_agents import SIUInvestigatorAgent, ClaimsAdjusterAgent, ClaimsManagerAgent

# Load environment
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_channel_id(channel_name="#claims-department"):
    """
    Get the Slack channel ID for the given channel name.
    Returns channel_id or None if not found.
    """
    try:
        slack_token = os.getenv("SLACK_MANAGER_BOT_TOKEN")
        if not slack_token:
            return None
        
        client = WebClient(token=slack_token)
        channel_name_clean = channel_name.lstrip('#')
        result = client.conversations_list(types="public_channel,private_channel")
        for channel in result["channels"]:
            if channel["name"] == channel_name_clean:
                return channel["id"]
        return None
    except Exception as e:
        print(f"⚠️  Error getting channel ID: {e}")
        return None
    
def send_message_to_slack(channel_id, message_text, agent_type="investigator"):
    """
    Send a message to Slack channel using the appropriate agent's bot token.
    
    Args:
        channel_id: The Slack channel ID to send the message to
        message_text: The message text to send
        agent_type: One of "investigator", "adjuster", or "manager"
    """
    # Map agent types to token names
    token_map = {
        "SIU_Investigator": ("SLACK_INVESTIGATOR_BOT_TOKEN",),
        "ClaimsAdjuster": ("SLACK_ADJUSTER_BOT_TOKEN",),
        "TransparencyAuditor": ("SLACK_AUDITOR_BOT_TOKEN",),
        "ClaimsManager": ("SLACK_MANAGER_BOT_TOKEN",),
    }
    
    if agent_type not in token_map:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be one of: {list(token_map.keys())}")
    
    bot_token_name = token_map[agent_type][0]
    slack_token = os.getenv(bot_token_name)
    
    if not slack_token:
        print(f"⚠️  {bot_token_name} not found, falling back to print statement")
        print(f"\n=== {agent_type.capitalize()} assessment ===\n{message_text}")
        return
    
    try:
        client = WebClient(token=slack_token)
        response = client.chat_postMessage(
            channel=channel_id,
            text=message_text
        )
        if response["ok"]:
            print(f"✅ Sent {agent_type} assessment to Slack")
        else:
            print(f"⚠️  Failed to send message to Slack: {response.get('error', 'Unknown error')}")
            print(f"\n=== {agent_type.capitalize()} assessment ===\n{message_text}")
    except SlackApiError as e:
        print(f"⚠️  Error sending message to Slack: {e.response['error']}")
        print(f"\n=== {agent_type.capitalize()} assessment ===\n{message_text}")
    except Exception as e:
        print(f"⚠️  Unexpected error sending to Slack: {e}")
        print(f"\n=== {agent_type.capitalize()} assessment ===\n{message_text}")

    
def wait_for_claim_from_slack(channel_name="#claims-department", bot_name="Claim Manager"):
    """
    Wait for a message from the Claim Manager bot in the specified Slack channel.
    Returns tuple: (message_text, channel_id)
    """
    slack_token = os.getenv("SLACK_MANAGER_BOT_TOKEN")
    slack_app_token = os.getenv("SLACK_MANAGER_APP_TOKEN")
    
    if not slack_token:
        raise ValueError("SLACK_MANAGER_BOT_TOKEN environment variable is required")
    if not slack_app_token:
        raise ValueError("SLACK_MANAGER_APP_TOKEN environment variable is required (for Socket Mode)")
    
    client = WebClient(token=slack_token)
    socket_client = SocketModeClient(app_token=slack_app_token, web_client=client)
    
    claim_message = None
    channel_id = None
    
    # Get channel ID
    try:
        # Remove # if present
        channel_name_clean = channel_name.lstrip('#')
        result = client.conversations_list(types="public_channel,private_channel")
        for channel in result["channels"]:
            if channel["name"] == channel_name_clean:
                channel_id = channel["id"]
                break
        
        if not channel_id:
            raise ValueError(f"Channel {channel_name} not found")
        
        print(f"✅ Found channel {channel_name} (ID: {channel_id})")
        print(f"⏳ Waiting for message from '{bot_name}' bot in {channel_name}...")
    except SlackApiError as e:
        raise Exception(f"Error getting channel info: {e.response['error']}")
    
    def process_message(client: SocketModeClient, req: SocketModeRequest):
        """Handle incoming Slack events"""
        nonlocal claim_message
        
        if req.type == "events_api":
            event = req.payload.get("event", {})
            event_type = event.get("type")
            
            if event_type == "message":
                # Skip message events that are not in our channel
                event_channel = event.get("channel")
                if event_channel != channel_id:
                    # Acknowledge but ignore
                    response = SocketModeResponse(envelope_id=req.envelope_id)
                    client.send_socket_mode_response(response)
                    return
                
                # Skip message edits, deletions, etc.
                if event.get("subtype") and event.get("subtype") not in ["bot_message", None]:
                    response = SocketModeResponse(envelope_id=req.envelope_id)
                    client.send_socket_mode_response(response)
                    return
                
                message_text = event.get("text", "")
                if not message_text:
                    response = SocketModeResponse(envelope_id=req.envelope_id)
                    client.send_socket_mode_response(response)
                    return
                
                # Check if message is from a bot
                bot_id = event.get("bot_id")
                if bot_id:
                    try:
                        bot_details = client.web_client.bots_info(bot=bot_id)
                        bot_username = bot_details.get("bot", {}).get("name", "").lower()
                        # Check if bot name matches (case-insensitive, partial match)
                        # Look for "Claim Manager", "Slack Manager", or "Claims Manager"
                        if (bot_name.lower() in bot_username or 
                            "claim manager" in bot_username or 
                            "slack manager" in bot_username or
                            "claims manager" in bot_username):
                            claim_message = message_text
                            print(f"✅ Received claim from {bot_name} bot: {message_text[:100]}...")
                            response = SocketModeResponse(envelope_id=req.envelope_id)
                            client.send_socket_mode_response(response)
                            socket_client.close()
                            return
                    except Exception as e:
                        print(f"⚠️  Error checking bot info: {e}")
                
                # Check if message is from a user (in case Claim Manager is a user, not a bot)
                user_id = event.get("user")
                if user_id and not bot_id:
                    try:
                        user_info = client.web_client.users_info(user=user_id)
                        user_name = (user_info.get("user", {}).get("real_name", "") or 
                                    user_info.get("user", {}).get("name", "")).lower()
                        # Check if user name matches
                        # Look for "Claim Manager", "Slack Manager", or "Claims Manager"
                        if (bot_name.lower() in user_name or 
                            "claim manager" in user_name or 
                            "slack manager" in user_name or
                            "claims manager" in user_name):
                            claim_message = message_text
                            print(f"✅ Received claim from {bot_name} user: {message_text[:100]}...")
                            response = SocketModeResponse(envelope_id=req.envelope_id)
                            client.send_socket_mode_response(response)
                            socket_client.close()
                            return
                    except Exception as e:
                        print(f"⚠️  Error checking user info: {e}")
                
                # Fallback: accept any message in the channel if no bot/user filtering works
                # This is useful for testing or if the bot name doesn't match exactly
                print(f"ℹ️  Received message in channel (not from {bot_name}), accepting as claim...")
                claim_message = message_text
                response = SocketModeResponse(envelope_id=req.envelope_id)
                client.send_socket_mode_response(response)
                socket_client.close()
                return
        
        # Acknowledge the event
        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)
    
    # Set up event handler
    socket_client.socket_mode_request_listeners.append(process_message)
    
    # Connect and wait
    socket_client.connect()
    print("🔌 Connected to Slack. Listening for messages...")
    
    # Wait for message (with timeout)
    timeout = 300  # 5 minutes
    start_time = time.time()
    while claim_message is None:
        if time.time() - start_time > timeout:
            socket_client.close()
            raise TimeoutError(f"No message received from {bot_name} within {timeout} seconds")
        time.sleep(0.5)
    
    return claim_message, channel_id

def parse_claim_text(text: str) -> InsuranceClaim:
    # Extract Policyholder
    claimant_match = re.search(r"Insured:\s*(.+)", text)
    claimant_name = claimant_match.group(1).strip() if claimant_match else "Unknown"

    # Extract claim type (look for Policy Type or type of claim in the narrative)
    claim_type_match = re.search(r"Policy Type:\s*(.+)", text)
    claim_type = claim_type_match.group(1).strip() if claim_type_match else "Unknown"

    # Extract claim amount from the Dwelling (A) + Personal Property (C) + ALE (D)
    amounts = re.findall(r"\$([\d,]+(?:\.\d{2})?)", text)
    total_amount = sum(float(a.replace(",", "")) for a in amounts) if amounts else 0.0

    # Generate a claim ID (or parse if included in text)
    claim_id_match = re.search(r"File:\s*#([A-Za-z0-9-]+)", text)
    claim_id = claim_id_match.group(1) if claim_id_match else f"SLACK-{uuid.uuid4()}"

    # Full text as description
    description = text.strip()

    return InsuranceClaim(
        claim_id=claim_id,
        claimant_name=claimant_name,
        claim_type=claim_type,
        claim_amount=total_amount,
        description=description,
    )
    
def print_claim_human_readable(claim: InsuranceClaim):
    print("="*60)
    print(f"Claim ID       : {claim.claim_id}")
    print(f"Claimant Name  : {claim.claimant_name}")
    print(f"Claim Type     : {claim.claim_type}")
    print(f"Claim Amount   : ${claim.claim_amount:,.2f}")
    print(f"Status         : {claim.status}")
    print("-"*60)
    print("Description:")
    print(claim.description)
    print("="*60)


def main(claim):
    """Main execution with insurance-specific agents and agent thoughts display"""
    
    print("🏗️  Initializing Insurance Claims System...\n")
    print("RECEIVED CLAIM")
    parsed_claim = parse_claim_text(claim)
    print_claim_human_readable(parsed_claim)
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
    
    claim = parsed_claim
    
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
        send_message_to_slack(channel_id, thought, agent_type=agent_name)
    print("-"*50)

    
    # Display final decision from Claims Manager
    print("\n⚖️ FINAL DECISION:")
    manager = orchestrator.agents["ClaimsManager"]
    final_decision = manager.final_decisions.get(claim.claim_id, "No decision recorded")
    print(final_decision)
    send_message_to_slack(channel_id, final_decision, agent_type="ClaimsManager")
    print("\n✅ Claim processing complete")
    
    return workflow, orchestrator

if __name__ == "__main__":
    claim = None
    while claim == None:
        try:
            claim, received_channel_id = wait_for_claim_from_slack(channel_name="#claims-department", bot_name="Claim Manager")
            if received_channel_id:
                channel_id = received_channel_id
            print(f"\n📝 Claim received: {claim[:100]}...\n")
        except Exception as e:
            print(f"❌ Error waiting for Slack message: {e}")
            print("⚠️  Skipping this claim, waiting for next one...\n")
            time.sleep(2)  # Brief pause before retrying
            continue
    orchestrator = main(claim)
    print("\n🎉 System demonstration complete!\n")
