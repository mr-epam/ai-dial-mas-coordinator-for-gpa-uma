import json
import os
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        ums_agent_endpoint: str,
        gpa_endpoint: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint
        self.gpa_endpoint = gpa_endpoint or os.environ.get("GPA_AGENT_ENDPOINT", "")

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        api_key = request.api_key
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )
        coord_stage = StageProcessor.open_stage(choice, "Coordination Request")
        coordination_request = await self.__prepare_coordination_request(client, request)
        coord_stage.append_content(coordination_request.model_dump_json())
        StageProcessor.close_stage_safely(coord_stage)

        agent_stage = StageProcessor.open_stage(choice, "Agent Response")
        agent_message = await self.__handle_coordination_request(
            coordination_request, choice, agent_stage, request
        )
        StageProcessor.close_stage_safely(agent_stage)
        return await self.__final_response(client, choice, request, agent_message)

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": CoordinationRequest.model_json_schema(),
            },
        }
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            extra_body={"response_format": response_format},
        )
        content = response.choices[0].message.content
        data = json.loads(content) if isinstance(content, str) else content
        return CoordinationRequest.model_validate(data)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            if getattr(msg, "role", None) == Role.USER:
                messages.append({"role": "user", "content": getattr(msg, "content", "") or ""})
            else:
                d = getattr(msg, "dict", None)
                if d and callable(d):
                    messages.append(msg.dict(exclude_none=True))
                else:
                    messages.append(msg.model_dump(exclude_none=True))
        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        additional = coordination_request.additional_instructions
        if coordination_request.agent_name == AgentName.GPA:
            if not self.gpa_endpoint:
                raise ValueError("GPA_AGENT_ENDPOINT or gpa_endpoint is required when routing to GPA")
            gateway = GPAGateway(self.gpa_endpoint)
        else:
            gateway = UMSAgentGateway(self.ums_agent_endpoint)
        return await gateway.response(choice, stage, request, additional)

    async def __final_response(
            self,
            client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)
        last_user_content = ""
        for msg in reversed(request.messages):
            if getattr(msg, "role", None) == Role.USER:
                last_user_content = getattr(msg, "content", "") or ""
                break
        agent_content = getattr(agent_message, "content", "") or ""
        augmented = f"Context from agent:\n{agent_content}\n\nUser request: {last_user_content}"
        messages.append({"role": "user", "content": augmented})

        stage = StageProcessor.open_stage(choice, "Final Response")
        content_parts: list[str] = []
        stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None) if chunk.choices else None
            if delta and getattr(delta, "content", None):
                part = delta.content
                content_parts.append(part)
                stage.append_content(part)
                choice.append_content(part)

        StageProcessor.close_stage_safely(stage)
        custom_content = getattr(agent_message, "custom_content", None)
        return Message(role=Role.ASSISTANT, content="".join(content_parts), custom_content=custom_content)
