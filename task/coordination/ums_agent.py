import json
from typing import Optional, Any

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice, CustomContent


_UMS_CONVERSATION_ID = "ums_conversation_id"


def _get_msg_state(msg: Any) -> Optional[dict]:
    """Get state dict from message custom_content if present."""
    content = getattr(msg, "custom_content", None)
    if content is None:
        return None
    if hasattr(content, "state"):
        return getattr(content, "state", None)
    if isinstance(content, dict):
        return content.get("state")
    return None


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint.rstrip("/")

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        ums_conv_id = self._get_ums_conversation_id(request)
        if not ums_conv_id:
            ums_conv_id = await self._create_ums_conversation()

        user_message = self._last_user_message(request)
        if additional_instructions:
            user_message = f"{additional_instructions.strip()}\n\n{user_message}".strip()
        content = await self._call_ums_agent(ums_conv_id, user_message, stage)

        custom_content = CustomContent(state={_UMS_CONVERSATION_ID: ums_conv_id})
        return Message(role=Role.ASSISTANT, content=content, custom_content=custom_content)

    def _get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists."""
        for msg in reversed(request.messages or []):
            if getattr(msg, "role", None) != Role.ASSISTANT:
                continue
            state = _get_msg_state(msg)
            if state and isinstance(state, dict) and _UMS_CONVERSATION_ID in state:
                val = state[_UMS_CONVERSATION_ID]
                return val if isinstance(val, str) else str(val)
        return None

    def _last_user_message(self, request: Request) -> str:
        """Return the last user message content (the current turn)."""
        for msg in reversed(request.messages or []):
            if getattr(msg, "role", None) == Role.USER:
                return getattr(msg, "content", "") or ""
        return ""

    async def _create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side."""
        url = f"{self.ums_agent_endpoint}/conversations"
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={})
            r.raise_for_status()
            data = r.json()
        return data["id"]

    async def _call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response."""
        url = f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat"
        body = {"message": {"role": "user", "content": user_message}, "stream": True}
        accumulated: list[str] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta") or {}
                        part = (delta.get("content") or "") if isinstance(delta, dict) else ""
                        if part:
                            accumulated.append(part)
                            stage.append_content(part)
        return "".join(accumulated)
