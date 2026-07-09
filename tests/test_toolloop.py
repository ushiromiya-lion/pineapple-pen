import threading
import time

from genio.core.toolloop import (
    LoopStatus,
    PendingInput,
    ToolError,
    ToolLoopSession,
    ToolSpec,
)
from google.genai import types
from toolloop_fakes import SIG, fc_part, model_turn, scripted


def _schema(schema_type: str, **kwargs) -> types.Schema:
    return types.Schema(type=schema_type, **kwargs)


def _object_schema(props: dict[str, types.Schema], required: list[str]) -> types.Schema:
    return types.Schema(type="OBJECT", properties=props, required=required)


def ping_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="ping",
        description="Ping.",
        parameters=_object_schema({"x": _schema("INTEGER")}, ["x"]),
    )


def finish_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="finish",
        description="Finish.",
        parameters=_object_schema({"note": _schema("STRING")}, ["note"]),
    )


def make_session(generate_fn, ping_handler=None) -> ToolLoopSession:
    def handle_ping(args: dict) -> dict:
        x = int(args["x"])
        if x < 0:
            raise ToolError("bad x")
        return {"pong": x}

    def handle_finish(args: dict) -> dict:
        return {"ok": True}

    return ToolLoopSession(
        model="fake",
        tools=[
            ToolSpec(ping_declaration(), ping_handler or handle_ping),
            ToolSpec(finish_declaration(), handle_finish, terminal=True),
        ],
        system_instruction="system",
        generate_fn=generate_fn,
    )


def function_responses(content: types.Content) -> list[dict]:
    return [part.function_response.response for part in content.parts]


def test_single_batch_happy_path():
    session = make_session(
        scripted([model_turn(fc_part("ping", {"x": 1}), fc_part("finish", {"note": "n"}))])
    )

    result = session.run_turn("go")

    assert result.completed is True
    assert result.model_turns == 1
    assert result.terminal_args == {"note": "n"}
    assert len(session._pending_responses) == 2
    assert [
        part.function_response.name for part in session._pending_responses
    ] == ["ping", "finish"]


def test_multi_step_composition():
    session = make_session(
        scripted(
            [
                model_turn(fc_part("ping", {"x": 1})),
                model_turn(fc_part("finish", {"note": "done"})),
            ]
        )
    )

    result = session.run_turn("go")

    assert result.completed is True
    assert [content.role for content in session.history] == [
        "user",
        "model",
        "user",
        "model",
    ]
    assert function_responses(session.history[2]) == [{"output": {"pong": 1}}]


def test_error_retry():
    session = make_session(
        scripted(
            [
                model_turn(fc_part("ping", {"x": -1})),
                model_turn(fc_part("ping", {"x": 1}), fc_part("finish", {"note": "ok"})),
            ]
        )
    )

    result = session.run_turn("go")

    assert result.completed is True
    assert function_responses(session.history[2])[0] == {"error": "bad x"}


def test_verbatim_history_preserves_signature():
    session = make_session(
        scripted([model_turn(fc_part("ping", {"x": 1}), fc_part("finish", {"note": "n"}))])
    )

    session.run_turn("go")

    model_contents = [content for content in session.history if content.role == "model"]
    assert model_contents
    for content in model_contents:
        for part in content.parts:
            if part.function_call:
                assert part.thought_signature == SIG


def test_unknown_tool_gets_error_response():
    session = make_session(
        scripted(
            [
                model_turn(fc_part("pong", {"x": 1})),
                model_turn(fc_part("finish", {"note": "ok"})),
            ]
        )
    )

    result = session.run_turn("go")

    assert result.completed is True
    assert function_responses(session.history[2])[0] == {"error": "unknown tool pong"}


def test_calls_after_terminal_in_same_batch_get_error():
    session = make_session(
        scripted([model_turn(fc_part("finish", {"note": "n"}), fc_part("ping", {"x": 1}))])
    )

    result = session.run_turn("go")

    assert result.completed is True
    assert function_responses(
        types.Content(role="user", parts=session._pending_responses)
    ) == [
        {"output": {"ok": True}},
        {"error": "resolution already committed; this call was ignored"},
    ]


def test_budget_exhaustion_forces_finish():
    seen_allowed = []

    def generate_fn(*, contents, config):
        fcc = config.tool_config.function_calling_config
        seen_allowed.append(fcc.allowed_function_names)
        if len(seen_allowed) <= 2:
            return model_turn(fc_part("ping", {"x": 1}))
        return model_turn(fc_part("finish", {"note": "forced"}))

    session = make_session(generate_fn)

    result = session.run_turn("go", max_model_turns=2)

    assert result.forced is True
    assert result.completed is True
    assert seen_allowed[-1] == ["finish"]


def test_forced_finish_noncompliance():
    session = make_session(
        scripted(
            [
                model_turn(fc_part("ping", {"x": 1})),
                model_turn(fc_part("ping", {"x": 2})),
                model_turn(fc_part("ping", {"x": 3})),
            ]
        )
    )

    result = session.run_turn("go", max_model_turns=2)

    assert result.forced is True
    assert result.completed is False


def test_pending_responses_flow_into_next_turn():
    session = make_session(
        scripted(
            [
                model_turn(fc_part("ping", {"x": 1}), fc_part("finish", {"note": "one"})),
                model_turn(fc_part("finish", {"note": "two"})),
            ]
        )
    )
    session.run_turn("first")

    session.run_turn("second")

    opening = session.history[2]
    assert opening.role == "user"
    assert opening.parts[0].function_response.name == "ping"
    assert opening.parts[1].function_response.name == "finish"
    assert opening.parts[2].text == "second"


def test_pending_input_suspends_and_resumes():
    pending_input = PendingInput({"kind": "pick"})

    def handle_ping(args: dict):
        return pending_input

    session = make_session(
        scripted([model_turn(fc_part("ping", {"x": 1}), fc_part("finish", {"note": "n"}))]),
        ping_handler=handle_ping,
    )
    result_holder = {}

    thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", session.run_turn("go"))
    )
    thread.start()
    deadline = time.monotonic() + 2
    while session.status != LoopStatus.AWAITING_INPUT and time.monotonic() < deadline:
        time.sleep(0.01)

    assert session.status == LoopStatus.AWAITING_INPUT
    assert session.pending_input.request == {"kind": "pick"}
    session.pending_input.fulfill({"card_ids": ["c1"]})
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result_holder["result"].completed is True
    assert session._pending_responses[0].function_response.response == {
        "output": {"card_ids": ["c1"]}
    }
