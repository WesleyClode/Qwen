from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None


class ChatMsgResponse(BaseModel):
    session_ids: List = []
    messages: List[List] = [[]]


res = ChatMsgResponse(session_ids=[1, 2, 3], messages=[["a", "b"], ["c", "d"], ["e", "f"]])

msg = ChatMessage(role="user", content="123")
print(msg)
print(f"\n{msg.role}:{msg.content}")
print(msg.content)

a = {"1": 2, "3": 4}
a_keys = list(a.keys())
a_vals = list(a.values())
for i in range(len(a_keys)):
    a_keys[i] = '`' + a_keys[i] + '`'
a_keys = ','.join(a_keys)
print(a_keys)
print(a_vals)
print("1" in a_keys)

sessions = res.session_ids
msgs = res.messages
for i in range(len(sessions)):
    print(f"{sessions[i]}: {msgs[i]}")
