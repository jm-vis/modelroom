"""LM Studio is not measured: the one sentence that says so, and the rule that picks what step 4 says.

The measuring switch stands on its negative branch (decided 2026-10-08, ground A-LS7 "proof open"). A request
to the id of a model that was loaded a moment ago, and unloaded since, may make LM Studio load it again --
just-in-time loading is on by default -- and the recording of that case, `Failed to load model ... Operation
canceled.`, proves neither a refusal nor a load. So modelroom sends the server no request that names a model.

There is therefore no load test here and none is imported: no body of a request, no warm-up, no run of
requests, no speed. This module is the single place the sentence is written, so the start screen, step 4,
the card and the README quote one text. A measurement, if a later sub-package proves one safe, is built
here and registers its own offer; until then `tests/test_loadtest_lmstudio.py` shows that no module of LM
Studio names the call a measurement would need.
"""

from __future__ import annotations

from .lmstudio_server import LmStudioModels

NOT_MEASURED_NOTE = "LM Studio is not measured: it may load a model on request"


def step_note(found: object) -> str | None:
    """What step 4 says for what the start screen read: the reason of a server that asks for a token as it
    came, the sentence for one that answered with a model list, nothing for the rest."""
    if isinstance(found, str):
        return found
    return NOT_MEASURED_NOTE if isinstance(found, LmStudioModels) else None


__all__ = ["NOT_MEASURED_NOTE", "step_note"]
