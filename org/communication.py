# org/communication.py
from typing import List, Dict
import datetime

class Message:
    def __init__(self, sender, receiver, content, task_id=None):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.task_id = task_id
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)

class CommunicationHub:
    def __init__(self):
        self.messages: List[Message] = []

    def send_message(self, sender, receiver, content, task_id=None):
        msg = Message(sender, receiver, content, task_id)
        self.messages.append(msg)
        receiver.receive_message(msg)
        return msg

    def broadcast(self, sender, receivers, content, task_id=None):
        msgs = []
        for r in receivers:
            msgs.append(self.send_message(sender, r, content, task_id))
        return msgs
