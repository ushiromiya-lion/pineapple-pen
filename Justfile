fix:
    ruff check --exit-zero --fix .
    ruff format .

clear:
    rm -r .cache

edit:
    uv run pyxel edit assets/sprites.pyxres

play:
    uv run python -m genio.main

ps:
    uv run python -m genio.main --edit

convert-videos:
    uv run python -m genio.gears.h264_encoder

# Use `pv` to rate-limit.
