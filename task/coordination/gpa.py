from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"
_GPA_DEPLOYMENT = "general-purpose-agent"


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert message to dict with exclude_none."""
    dump = getattr(msg, "model_dump", None)
    if dump and callable(dump):
        return dump(exclude_none=True)
    d = getattr(msg, "dict", None)
    if d and callable(d):
        return d(exclude_none=True)
    return dict(msg)


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


def _to_sdk_attachment(att: Any) -> Attachment:
    """Convert client Attachment to SDK Attachment."""
    raw = getattr(att, "model_dump", None)
    if raw and callable(raw):
        data = raw(exclude_none=True)
    else:
        data = getattr(att, "dict", lambda **kw: {})(exclude_none=True)
    return Attachment(**data)


def _add_attachments_to_stage(stage: Stage, attachments: list[Any]) -> None:
    """Add attachments to a stage if the API supports it."""
    for att in attachments:
        try:
            sdk_att = _to_sdk_attachment(att)
        except Exception:
            continue
        add = getattr(stage, "add_attachment", None)
        if add and callable(add):
            add(sdk_att)
        else:
            attrs = getattr(stage, "attachments", None)
            if attrs is not None and isinstance(attrs, list):
                attrs.append(sdk_att)


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        messages = self._prepare_gpa_messages(request, additional_instructions)
        headers = {}
        req_headers = getattr(request, "headers", None)
        if req_headers is not None:
            cid = getattr(req_headers, "get", lambda k: None)("x-conversation-id")
            if cid:
                headers["x-conversation-id"] = cid

        client = AsyncDial(base_url=self.endpoint, api_version="2025-01-01-preview")
        extra = {"extra_headers": headers} if headers else {}
        stream = await client.chat.completions.create(
            model=_GPA_DEPLOYMENT,
            messages=messages,
            stream=True,
            **extra,
        )

        content_parts: list[str] = []
        result_attachments: list[Attachment] = []
        result_state: dict[str, Any] = {}
        stages_map: dict[int, Stage] = {}

        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None) if chunk.choices else None
            if delta is None:
                continue
            part = getattr(delta, "content", None)
            if part:
                content_parts.append(part)
                stage.append(part)
            cc = getattr(delta, "custom_content", None)
            if cc is None:
                continue
            atts = getattr(cc, "attachments", None) or []
            if atts:
                for att in atts:
                    try:
                        result_attachments.append(_to_sdk_attachment(att))
                    except Exception:
                        pass
            st = getattr(cc, "state", None)
            if st and isinstance(st, dict):
                result_state.update(st)
            cc_dict = None
            if hasattr(cc, "model_dump") and callable(cc.model_dump):
                cc_dict = cc.model_dump(exclude_none=True)
            elif hasattr(cc, "dict") and callable(cc.dict):
                cc_dict = cc.dict(exclude_none=True)
            else:
                cc_dict = {}
            if not isinstance(cc_dict, dict):
                continue
            for stg in cc_dict.get("stages") or []:
                idx = stg.get("index")
                if idx is None:
                    continue
                if idx in stages_map:
                    s = stages_map[idx]
                else:
                    s = StageProcessor.open_stage(choice, stg.get("name"))
                    stages_map[idx] = s
                if stg.get("content"):
                    s.append(stg["content"])
                _add_attachments_to_stage(s, stg.get("attachments") or [])
                if stg.get("status") == "completed":
                    StageProcessor.close_stage_safely(s)

        coordinator_state = {_IS_GPA: True, _GPA_MESSAGES: result_state}
        custom_content = CustomContent(state=coordinator_state, attachments=result_attachments)
        return Message(role=Role.ASSISTANT, content="".join(content_parts), custom_content=custom_content)

    def _prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages: list[dict[str, Any]] = []
        messages = request.messages or []
        for i in range(len(messages)):
            msg = messages[i]
            if getattr(msg, "role", None) != Role.ASSISTANT:
                continue
            state = _get_msg_state(msg)
            if not state or not isinstance(state, dict) or not state.get(_IS_GPA) or _GPA_MESSAGES not in state:
                continue
            if i > 0:
                res_messages.append(_msg_to_dict(messages[i - 1]))
            copied = deepcopy(msg)
            gpa_state = state[_GPA_MESSAGES]
            gpa_content = CustomContent(state=gpa_state) if gpa_state else None
            copied_dict = _msg_to_dict(copied)
            if gpa_content is not None:
                copied_dict["custom_content"] = (
                    gpa_content.model_dump(exclude_none=True)
                    if hasattr(gpa_content, "model_dump")
                    else {"state": gpa_state}
                )
            res_messages.append(copied_dict)

        last_user = None
        for m in reversed(messages):
            if getattr(m, "role", None) == Role.USER:
                last_user = m
                break
        if last_user is not None:
            last_dict = _msg_to_dict(last_user)
            if additional_instructions:
                prev = last_dict.get("content") or ""
                last_dict["content"] = f"{additional_instructions.strip()}\n\n{prev}".strip()
            res_messages.append(last_dict)

        return res_messages
