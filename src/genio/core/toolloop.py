from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from google.genai import types
from structlog import get_logger

logger = get_logger()


class ToolError(Exception):
    """Raised by a tool handler for a model-repairable error.

    The message is sent back to the model as {"error": message}.
    """


class PendingInput:
    """Suspend the loop until an external party supplies the tool result."""

    def __init__(self, request: Any) -> None:
        self.request = request
        self._event = threading.Event()
        self._value: dict | None = None

    def fulfill(self, value: dict) -> None:
        self._value = value
        self._event.set()

    def wait(self) -> dict:
        self._event.wait()
        assert self._value is not None
        return self._value


@dataclass(frozen=True)
class ToolSpec:
    declaration: types.FunctionDeclaration
    handler: Callable[[dict], dict | PendingInput]
    terminal: bool = False


class LoopStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    AWAITING_INPUT = auto()
    DONE = auto()


@dataclass
class TurnResult:
    completed: bool
    terminal_args: dict | None = None
    terminal_output: dict | None = None
    model_turns: int = 0
    forced: bool = False


@dataclass
class _CallBatchResult:
    response_parts: list[types.Part]
    terminal_args: dict | None = None
    terminal_output: dict | None = None
    terminal_succeeded: bool = False


@dataclass
class ToolLoopSession:
    """One manual Gemini tool-calling conversation."""

    model: str
    tools: list[ToolSpec]
    system_instruction: str
    thinking_level: str = "minimal"
    safety_settings: list[types.SafetySetting] | None = None
    generate_fn: Callable[..., types.GenerateContentResponse] | None = None

    history: list[types.Content] = field(default_factory=list)
    status: LoopStatus = LoopStatus.IDLE
    pending_input: PendingInput | None = None
    _pending_responses: list[types.Part] = field(default_factory=list)

    def __post_init__(self) -> None:
        terminal_tools = [tool for tool in self.tools if tool.terminal]
        if len(terminal_tools) != 1:
            raise ValueError("ToolLoopSession requires exactly one terminal tool")

    def _tool_by_name(self, name: str) -> ToolSpec | None:
        for tool in self.tools:
            if tool.declaration.name == name:
                return tool
        return None

    def _terminal_name(self) -> str:
        terminal_tools = [tool for tool in self.tools if tool.terminal]
        assert len(terminal_tools) == 1
        name = terminal_tools[0].declaration.name
        assert name is not None
        return name

    def _config(self, allowed: list[str] | None) -> types.GenerateContentConfig:
        fcc = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=allowed,
        )
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[
                types.Tool(
                    function_declarations=[tool.declaration for tool in self.tools]
                )
            ],
            tool_config=types.ToolConfig(function_calling_config=fcc),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            safety_settings=self.safety_settings,
        )

    def _generate(
        self, allowed: list[str] | None = None
    ) -> types.GenerateContentResponse:
        config = self._config(allowed)
        if self.generate_fn is not None:
            return self.generate_fn(contents=self.history, config=config)
        from genio.core.llm import default_llm

        return default_llm().client.models.generate_content(
            model=self.model, contents=self.history, config=config
        )

    def _payload_from_output(self, output: dict) -> dict:
        if "output" in output or "error" in output:
            return output
        return {"output": output}

    def _call_response(self, name: str, payload: dict) -> types.Part:
        return types.Part.from_function_response(name=name, response=payload)

    def _invoke_tool(self, spec: ToolSpec, args: dict) -> dict:
        output = spec.handler(args)
        if isinstance(output, PendingInput):
            self.pending_input = output
            self.status = LoopStatus.AWAITING_INPUT
            output = output.wait()
            self.pending_input = None
            self.status = LoopStatus.RUNNING
        return self._payload_from_output(output)

    def _handle_calls(self, calls: list[types.FunctionCall]) -> _CallBatchResult:
        response_parts = []
        terminal_name = self._terminal_name()
        terminal_handled = False
        terminal_args = None
        terminal_output = None
        terminal_succeeded = False

        for call in calls:
            name = call.name or ""
            if terminal_handled:
                payload = {
                    "error": "resolution already committed; this call was ignored"
                }
            else:
                spec = self._tool_by_name(name)
                if spec is None:
                    payload = {"error": f"unknown tool {name}"}
                else:
                    try:
                        payload = self._invoke_tool(spec, dict(call.args or {}))
                    except ToolError as exc:
                        payload = {"error": str(exc)}
                    if spec.terminal and "error" not in payload:
                        terminal_handled = True
                        terminal_succeeded = True
                        terminal_args = dict(call.args or {})
                        terminal_output = payload.get("output", payload)
                    elif name == terminal_name:
                        terminal_handled = True
            response_parts.append(self._call_response(name, payload))

        return _CallBatchResult(
            response_parts=response_parts,
            terminal_args=terminal_args,
            terminal_output=terminal_output,
            terminal_succeeded=terminal_succeeded,
        )

    def _all_errors(self, parts: list[types.Part]) -> bool:
        if not parts:
            return False
        for part in parts:
            response = part.function_response.response
            if "error" not in response:
                return False
        return True

    def run_turn(
        self,
        message: str,
        *,
        max_model_turns: int = 6,
        max_error_turns: int = 2,
    ) -> TurnResult:
        self.status = LoopStatus.RUNNING
        pending = self._pending_responses
        self._pending_responses = []
        opening_parts = [*pending, types.Part(text=message)]
        self.history.append(types.Content(role="user", parts=opening_parts))

        consecutive_error_turns = 0
        terminal_name = self._terminal_name()

        for turn in range(1, max_model_turns + 1):
            resp = self._generate()
            self.history.append(resp.candidates[0].content)
            calls = resp.function_calls or []
            if not calls:
                self.history.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "You must respond with tool calls. "
                                    f"Call {terminal_name} to end the resolution."
                                )
                            )
                        ],
                    )
                )
                consecutive_error_turns += 1
                if consecutive_error_turns >= max_error_turns:
                    break
                continue

            batch = self._handle_calls(calls)
            if batch.terminal_succeeded:
                self._pending_responses = batch.response_parts
                self.status = LoopStatus.DONE
                return TurnResult(
                    completed=True,
                    terminal_args=batch.terminal_args,
                    terminal_output=batch.terminal_output,
                    model_turns=turn,
                )

            self.history.append(
                types.Content(role="user", parts=batch.response_parts)
            )
            if self._all_errors(batch.response_parts):
                consecutive_error_turns += 1
            else:
                consecutive_error_turns = 0
            if consecutive_error_turns >= max_error_turns:
                break

        self.history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"Stop. Call {terminal_name} now to commit what is valid."
                    )
                ],
            )
        )
        resp = self._generate(allowed=[terminal_name])
        self.history.append(resp.candidates[0].content)
        calls = resp.function_calls or []
        terminal_calls = [call for call in calls if call.name == terminal_name]
        if terminal_calls:
            batch = self._handle_calls(calls)
            if batch.terminal_succeeded:
                self._pending_responses = batch.response_parts
                self.status = LoopStatus.DONE
                return TurnResult(
                    completed=True,
                    terminal_args=batch.terminal_args,
                    terminal_output=batch.terminal_output,
                    model_turns=max_model_turns + 1,
                    forced=True,
                )

        self.status = LoopStatus.DONE
        return TurnResult(
            completed=False,
            model_turns=max_model_turns + 1,
            forced=True,
        )
