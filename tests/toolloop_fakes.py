from collections.abc import Callable

from google.genai import types

SIG = b"skip_thought_signature_validator"


def fc_part(name: str, args: dict) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(name=name, args=args),
        thought_signature=SIG,
    )


def model_turn(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=list(parts)))
        ]
    )


def scripted(responses: list) -> Callable:
    it = iter(responses)

    def generate_fn(*, contents, config):
        return next(it)

    return generate_fn
