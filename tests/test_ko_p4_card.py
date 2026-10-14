"""P4, part 3: the setup note of a model card (AP9-KO, V7, decision E4-B) -- the finite rule of recognition with positive
and negative fixtures, and its use in the catalog run. The cards are invented after the real pattern: an ordinary start
block, an explicit requirement of a package for inference, a card with several models, negation, training and the
instructions of another model. Nothing of a card is run or copied; a test below proves the rule starts no process."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta

import pytest

import fixture_ko_p4 as F
import catalog_scan_card as C
import catalog_scan_packages as K
from test_catalog_scan import NOW, TODAY, Env, Hub, scanner

from modelroom.catalog_findings import CARD_MAX_BYTES, SetupNote

MODEL = "acme/Nova-7B"

START = "```bash\nvllm serve acme/Nova-7B --max-model-len 8192\n```\n"
REQUIRES = "Nova requires the **nova-inference** package that provides the Nova vLLM plugin (`pip install 'nova-inference>=1'`).\n"


def card(*parts: str) -> str:
    return "# Nova-7B\n\nA model.\n\n" + "\n".join(parts)


# --- positive: what the rule recognizes -------------------------------------------------------------------------------------


def test_an_ordinary_start_block_of_this_model_is_instructions_without_a_package():
    assert C.detect(card("## Serving\n\n" + START), MODEL) == C.Detection("instructions", None)


def test_an_explicit_requirement_of_a_package_for_inference_names_the_package():
    assert C.detect(card(REQUIRES, START), MODEL) == C.Detection("instructions", "nova-inference")


@pytest.mark.parametrize("sentence", [
    "Nova requires the `nova-inference` package for inference with vLLM.",
    "This model requires the **nova-inference** package to serve it with vLLM.",
    "- It needs the nova-inference python package for vLLM inference.",
    "Nova-7B requires the nova-inference package, which provides the vLLM plugin.",
    "The model needs the **nova-inference** package; vLLM loads its plugin.",
])
def test_the_forms_of_an_explicit_inference_requirement(sentence):
    assert C.detect(card(sentence, START), MODEL).package == "nova-inference"


def test_the_start_line_may_have_a_prompt_continuation_lines_and_another_case():
    block = "```sh\n$ vllm serve \\\n    ACME/nova-7b \\\n    --kv-cache-dtype fp8\n```\n"
    assert C.detect(card(block), MODEL).status == "instructions"


def test_a_block_without_a_language_and_one_of_a_shell_language_count_the_same():
    assert C.detect(card("```\nvllm serve acme/Nova-7B\n```\n"), MODEL).status == "instructions"
    assert C.detect(card("~~~console\nvllm serve acme/Nova-7B\n~~~\n"), MODEL).status == "instructions"


def test_a_card_with_blocks_for_several_models_is_instructions_when_one_of_them_is_this_model():
    other = "```bash\nvllm serve other/Another-1\n```\n"
    assert C.detect(card(other, START), MODEL).status == "instructions"


def test_the_front_matter_is_not_the_text_of_the_card():
    text = "---\nlicense: apache-2.0\ntags: [vllm serve acme/Nova-7B]\n---\n# Nova-7B\n"
    assert C.detect(text, MODEL) == C.Detection("none_recognized", None)


# --- negative: what it must not say ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", [
    "Nova does not require the **nova-inference** package for vLLM.",
    "Nova no longer needs the nova-inference package for inference with vLLM.",
    "Nova never requires the nova-inference package when it is served with vLLM.",
    "You can serve Nova with vLLM without the nova-inference package.",
])
def test_a_negation_gives_no_package_name(sentence):
    assert C.detect(card(sentence, START), MODEL) == C.Detection("instructions", None)


def test_a_training_or_conversion_requirement_gives_no_package_name_by_the_sentence_and_by_the_heading():
    by_sentence = "Fine-tuning Nova requires the **nova-train** package."
    by_heading = "## Training\n\nNova requires the nova-extra package for vLLM serving.\n"
    assert C.detect(card(by_sentence, START), MODEL).package is None
    assert C.detect(card(by_heading, START), MODEL).package is None
    after = "## Training\n\nText.\n\n## Inference\n\nNova requires the **nova-inference** package for vLLM.\n"
    assert C.detect(card(after, START), MODEL).package == "nova-inference"


@pytest.mark.parametrize("sentence", [
    "Nova-Large requires the **big-inference** package for vLLM.",
    "Orbit requires the orbit-inference package for vLLM.",
    "The Nova-Large variant needs the big-inference package for vLLM inference.",
    "Nova 2 requires the nova2-inference package for vLLM.",
])
def test_the_instructions_for_another_model_give_no_package_name(sentence):
    assert C.detect(card(sentence, START), MODEL).package is None


@pytest.mark.parametrize("sentence", [
    "To serve Other-7B, it requires the **other-engine** package for vLLM.",
    "To run Orbit with vLLM, this model needs the orbit-engine package.",
    "To use the other engine in vLLM, Nova requires the **other-engine** package.",
])
def test_a_lead_in_that_names_another_model_gives_no_package_name(sentence):
    assert C.detect(card(sentence, START), MODEL).package is None


@pytest.mark.parametrize("lead", ["To serve this model with vLLM, ", "To serve Nova, ", "To run Nova-7B in vLLM, ", "To deploy it using vLLM, "])
def test_a_lead_in_that_names_this_model_or_nothing_keeps_the_name(lead):
    assert C.detect(card(lead + "it requires the **nova-inference** package for vLLM.", START), MODEL).package == "nova-inference"


@pytest.mark.parametrize("text", [
    "Nova-7B requires the nova-extra package for fine tuning with vLLM.\n",
    "Nova-7B requires the nova-extra package for tuning with vLLM.\n",
    "## Fine tuning\n\nNova-7B requires the nova-extra package for vLLM.\n",
    "## Tuning\n\nNova-7B requires the nova-extra package for vLLM.\n",
])
def test_fine_tuning_with_a_space_and_tuning_alone_are_training_too(text):
    assert C.detect(card(text, START), MODEL).package is None


def test_a_subheading_does_not_lift_the_training_context_of_its_section():
    nested = "## Training\n\n### Dependencies\n\nNova requires the nova-extra package for vLLM serving.\n"
    assert C.detect(card(nested, START), MODEL).package is None
    sibling = "## Training\n\ntext\n\n## Dependencies\n\nNova requires the **nova-inference** package for vLLM.\n"
    assert C.detect(card(sibling, START), MODEL).package == "nova-inference"
    deeper = "### Training\n\n#### Setup\n\n##### Pins\n\nNova requires the nova-extra package for vLLM serving.\n\n### Inference\n\nNova requires the **nova-inference** package for vLLM.\n"
    assert C.detect(card(deeper, START), MODEL).package == "nova-inference"


def test_two_different_packages_are_ambiguous_and_give_none():
    both = "Nova requires the **nova-inference** package for vLLM.\n\nNova requires the **nova-plugin** package for vLLM inference.\n"
    assert C.detect(card(both, START), MODEL) == C.Detection("instructions", None)


def test_the_same_package_named_twice_is_one_package():
    twice = "Nova requires the **nova-inference** package for vLLM.\n\nTo serve, Nova needs the nova-inference package with vLLM.\n"
    assert C.detect(card(twice, START), MODEL).package == "nova-inference"


@pytest.mark.parametrize("text", [
    "```bash\npip install 'nova-inference>=1'\n```\n",  # an install line is no statement
    "Install nova-inference to run it.\n",
    "Nova requires the **nova-inference** package.\n",  # no word of inference or vLLM besides the name
    "Nova optionally uses the nova-inference package with vLLM.\n",
    "Nova requires the **nova inference; rm -rf /** package for vLLM.\n",
    "Nova requires the **-bad** package for vLLM.\n",
])
def test_forms_outside_the_rule_give_no_package_name_and_no_claim_that_nothing_is_required(text):
    found = C.detect(card(text, START), MODEL)
    assert found == C.Detection("instructions", None)
    assert C.detect(card(text), MODEL) == C.Detection("none_recognized", None)  # without the block: nothing recognized, said as that


def test_a_requirement_inside_a_code_block_is_not_a_sentence_of_the_card():
    block = "```text\nNova requires the nova-inference package for vLLM.\n```\n"
    assert C.detect(card(block, START), MODEL).package is None


def test_an_inline_mention_or_a_block_for_another_model_is_no_start_block_of_this_model():
    assert C.detect(card("Run `vllm serve acme/Nova-7B` to start.\n"), MODEL).status == "none_recognized"
    assert C.detect(card("```bash\nvllm serve other/Another-1\n```\n"), MODEL).status == "none_recognized"
    assert C.detect(card("```bash\nvllm serve acme/Nova-7B-FP8\n```\n"), MODEL).status == "none_recognized"
    assert C.detect(card("```bash\nvllm serve ./local/acme/Nova-7B\n```\n"), MODEL).status == "none_recognized"


def test_a_requirement_without_a_start_block_is_not_stored_as_instructions():
    assert C.detect(card(REQUIRES), MODEL) == C.Detection("none_recognized", None)


def test_an_empty_card_recognizes_nothing():
    assert C.detect("", MODEL) == C.Detection("none_recognized", None)


@pytest.mark.parametrize("text", [
    "## Other-7B\n\nIt requires the **other-engine** package for vLLM.\n",  # a pronoun under the heading of another model
    "## Other-7B\n\n### Serving\n\nThis model needs the other-engine package for vLLM inference.\n",  # ... under a subheading of it
    "Other-7B is our larger sibling. It requires the other-engine package for vLLM.\n",  # another model earlier in the paragraph
    "Other-7B is our larger sibling.\n\nIt requires the other-engine package for vLLM.\n",  # ... in another paragraph of the section
    "It requires the other-engine package for vLLM, as other/Another-1 does.\n",  # ... in the sentence itself
    "## Variants\n\n- Other-7B: the larger one\n- It needs the other-engine package for vLLM inference.\n",  # a list of several models
    "## Nova 70B\n\nIt requires the other-engine package for vLLM.\n",  # a size names another model too
    "## Nova-70B\n\nNova requires the other-engine package for vLLM.\n",  # the leading word of the name is no name of this model
    "| Model | Note |\n|---|---|\n| Other-7B | ok |\n| x | It requires the other-engine package for vLLM. |\n",  # a table
    "| It requires the other-engine package for vLLM. |\n",  # a table row is no sentence of running text
])
def test_a_pronoun_or_the_leading_word_near_another_model_gives_no_package_name(text):
    assert C.detect(card(text, START), MODEL) == C.Detection("instructions", None)


@pytest.mark.parametrize("text", [
    "## Other-7B\n\nText about it.\n\n## Serving\n\nIt requires the **nova-inference** package for vLLM.\n",  # a later section of no other model
    "## Other-7B\n\nNova-7B requires the nova-inference package for vLLM.\n",  # the whole name of this model is no pronoun
    "## Serving Nova-7B\n\nIt requires the nova-inference package for vLLM 0.6 or newer, see https://example.org/x-1.2/doc.\n",
])
def test_a_reference_that_can_only_mean_this_model_keeps_the_name(text):
    assert C.detect(card(text, START), MODEL).package == "nova-inference"


def test_the_rule_runs_no_process_and_reads_no_file(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a card must not start anything")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "popen", forbidden)
    assert C.detect(card(REQUIRES, "```bash\nvllm serve acme/Nova-7B; touch /tmp/x\n$(rm -rf /)\n```\n"), MODEL).status == "none_recognized"


# --- the note a read makes ---------------------------------------------------------------------------------------------------


def test_the_note_is_bound_to_model_card_revision_address_and_day():
    note = C.note_of("acme/Nova-7B", C.Detection("instructions", "nova-inference"), F.SHA_A, TODAY)
    assert isinstance(note, SetupNote)
    assert (note.status, note.package, note.card_revision, note.checked.isoformat()) == ("instructions", "nova-inference", F.SHA_A, TODAY)
    assert note.source_url == f"https://huggingface.co/acme/Nova-7B/blob/{F.SHA_A}/README.md"


def test_a_card_that_was_not_read_is_a_note_that_says_why_in_one_ascii_line():
    note = C.not_read("acme/Nova-7B", "the card is longer than 131072 bytes: ä" + "x" * 200, TODAY)
    assert note.status == "not_read" and note.card_revision is None and note.source_url == "https://huggingface.co/acme/Nova-7B"
    assert note.reason.isascii() and len(note.reason) <= 97


# --- in the run -----------------------------------------------------------------------------------------------------------------

ROW = "acme/Atlas-2-9B"
ATLAS_START = "```bash\nvllm serve acme/Atlas-2-9B --max-model-len 8192\n```\n"
ATLAS_REQUIRES = "Atlas requires the **atlas-inference** package that provides the Atlas vLLM plugin.\n"


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    document = json.loads(e.catalog.read_bytes())
    document["schema_version"] = 3
    e.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return e


def run(env, hub, *, now=NOW, **kw):
    transport = F.Recording(hub.m)
    result = scanner.scan(env.catalog, env.rejected, env.out, transport=transport, now=now, budget=300, sub_budget=30, packages=True, **kw)
    day = now.date().isoformat()
    return json.loads((env.out / day / "proposal.json").read_bytes()), result, transport


def note_change(proposal, repo=ROW):
    return next((c for c in proposal["changes"] if c["change"] == "field" and c["field"] == "setup_note" and c["hf_repo"] == repo), None)


def store_note(env, note):
    document = json.loads(env.catalog.read_bytes())
    for row in document["models"]:
        if row["hf_repo"] == ROW:
            row["setup_note"] = note
    env.catalog.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def stored_note(**over):
    base = {"model": ROW, "status": "instructions", "card_revision": F.SHA_B, "source_url": f"https://huggingface.co/{ROW}/blob/{F.SHA_B}/README.md",
            "checked": "2026-09-01", "package": None, "reason": None}
    base.update(over)
    return base


def test_a_card_with_a_start_block_and_a_requirement_is_proposed_as_a_setup_note_of_that_revision(env):
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, "# Atlas\n\n" + ATLAS_REQUIRES + "\n" + ATLAS_START, F.SHA_A)
    proposal, _, transport = run(env, hub)
    change = note_change(proposal)
    assert change["stage"] == "A9" and change["fields"]["setup_note"]["old"] is None
    assert change["fields"]["setup_note"]["new"] == {
        "model": ROW, "status": "instructions", "card_revision": F.SHA_A, "source_url": f"https://huggingface.co/{ROW}/blob/{F.SHA_A}/README.md",
        "checked": TODAY, "package": "atlas-inference", "reason": None}
    card_calls = [i for i, (_, url) in enumerate(transport.calls) if url == K.card_url(ROW, F.SHA_A)]
    assert len(card_calls) == 1 and transport.limits[card_calls[0]] == CARD_MAX_BYTES
    assert not any(url.endswith("/resolve/main/README.md") for _, url in transport.calls)


def test_a_card_without_a_statement_is_no_proposal_when_there_was_no_note(env):
    hub = Hub()
    F.quiet_world(hub)
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None


def test_the_revision_is_fixed_first_and_a_card_from_another_revision_is_not_taken(env):
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, "# Atlas\n\n" + ATLAS_START, F.SHA_A)
    F.put(hub, K.card_url(ROW, F.SHA_A), ("# Atlas\n\n" + ATLAS_START).encode(), headers={"X-Repo-Commit": F.SHA_B})
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None
    assert any(ROW in note and "not read" in note for note in proposal["notes"])


@pytest.mark.parametrize("how", ["404", "500", "too long", "no utf8", "model unreadable", "model no sha", "redirect off host"])
def test_a_card_that_was_not_read_changes_no_stored_note_and_is_reported_not_as_no_hit(env, how):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, "x", F.SHA_A)
    url = K.card_url(ROW, F.SHA_A)
    if how == "404":
        F.put(hub, url, b"", status=404)
    elif how == "500":
        F.put(hub, url, b"", status=500)
    elif how == "too long":
        F.put(hub, url, b"a" * (CARD_MAX_BYTES + 1), headers={"X-Repo-Commit": F.SHA_A})
    elif how == "no utf8":
        F.put(hub, url, b"\xff\xfe\x00", headers={"X-Repo-Commit": F.SHA_A})
    elif how == "model unreadable":
        F.put(hub, K.model_url(ROW), b"", status=503)
    elif how == "model no sha":
        F.put(hub, K.model_url(ROW), {"id": ROW})
    else:
        F.put(hub, url, b"", status=302, headers={"Location": "https://evil.example.org/README.md"})
    before = env.catalog.read_bytes()
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None and env.catalog.read_bytes() == before
    assert any(note.startswith(f"{ROW}: model card not read") and "setup note is unchanged" in note for note in proposal["notes"])


def test_a_card_of_the_stored_revision_is_not_read_again(env):
    store_note(env, stored_note(card_revision=F.SHA_A, source_url=f"https://huggingface.co/{ROW}/blob/{F.SHA_A}/README.md"))
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, None, F.SHA_A)
    proposal, result, transport = run(env, hub)
    assert note_change(proposal) is None and result.exit_code == 0
    assert not any(url == K.card_url(ROW, F.SHA_A) for _, url in transport.calls)


def test_a_note_of_another_card_revision_is_read_again_and_replaced_by_what_the_card_says_now(env):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, "# Atlas\n\n" + ATLAS_START, F.SHA_A)
    proposal, _, _ = run(env, hub)
    new = note_change(proposal)["fields"]["setup_note"]["new"]
    assert (new["card_revision"], new["status"], new["package"]) == (F.SHA_A, "instructions", None)


def test_a_card_that_no_longer_has_instructions_replaces_the_note_by_the_statement_that_nothing_was_recognized(env):
    store_note(env, stored_note())
    hub = Hub()
    F.quiet_world(hub)
    F.card(hub, ROW, "# Atlas\n\nNothing.\n", F.SHA_A)
    proposal, _, _ = run(env, hub)
    new = note_change(proposal)["fields"]["setup_note"]["new"]
    assert (new["status"], new["card_revision"], new["package"]) == ("none_recognized", F.SHA_A, None)


def test_the_card_of_a_model_that_is_not_a_text_or_vision_row_is_not_asked(env):
    hub = Hub()
    F.quiet_world(hub)
    _, _, transport = run(env, hub)
    assert not any("Comet-Embed" in url or "Comet-Echo" in url for _, url in transport.calls)


# --- the start of a card longer than the window (GF decision a) ---------------------------------------------------------------

FILLER = "".join(f"| row {i} | " + "ä" * 120 + " |\n" for i in range(2000))  # bloated HTML-like tables, far over the window


def big_card(hub, text: str, commit: str = F.SHA_A) -> bytes:
    """A card longer than the window: the Hub answers the range with its first `CARD_MAX_BYTES` bytes (206)."""
    raw = text.encode("utf-8")
    assert len(raw) > CARD_MAX_BYTES
    ranged(hub, raw[:CARD_MAX_BYTES], f"bytes 0-{CARD_MAX_BYTES - 1}/{len(raw)}", commit)
    return raw


def ranged(hub, body: bytes, content_range: str | None, commit: str = F.SHA_A) -> None:
    """The model answer and a `206` for the card with `content_range` (None: the header is missing)."""
    F.card(hub, ROW, None, F.SHA_A)
    headers = {"X-Repo-Commit": commit} | ({"Content-Range": content_range} if content_range is not None else {})
    F.put(hub, K.card_url(ROW, F.SHA_A), body, status=206, headers=headers)


@pytest.mark.parametrize("body, content_range", [
    (b"# Atlas\n\nNothing.\n", "bytes 0-17/200000"),  # short, of a longer card: no proof of a whole card
    (b"# Atlas\n\nNothing.\n", None),  # no range said
    (b"# Atlas\n\nNothing.\n", "bytes 0-9/18"),  # a range that is not the body
    (b"# Atlas\n\nNothing.\n", "bytes 5-22/40"),  # not from the start
    (b"# Atlas\n\nNothing.\n", "bytes 0-17/x"),  # no total
], ids=["short of a longer card", "no range", "range not the body", "not from the start", "no total"])
def test_a_206_whose_range_does_not_prove_a_whole_card_or_a_start_is_not_read(env, body, content_range):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    ranged(hub, body, content_range)
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None and any(note.startswith(f"{ROW}: model card not read") for note in proposal["notes"])


def test_a_206_of_a_whole_card_of_exactly_the_window_is_the_whole_card(env):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    body = ("# Atlas\n\n" + "x" * CARD_MAX_BYTES).encode("utf-8")[: CARD_MAX_BYTES - 1] + b"\n"
    ranged(hub, body, f"bytes 0-{CARD_MAX_BYTES - 1}/{CARD_MAX_BYTES}")
    proposal, _, _ = run(env, hub)
    assert note_change(proposal)["fields"]["setup_note"]["new"]["status"] == "none_recognized"


def test_a_statement_still_open_at_the_end_of_the_window_gives_no_package_name(env):
    hub = Hub()
    F.quiet_world(hub)
    head = "# Atlas\n\n" + ATLAS_START + "\n"
    tail = "Atlas requires the atlas-inference package for vLLM\n"
    pad = CARD_MAX_BYTES - len((head + "\n\n" + tail).encode("utf-8"))
    raw = big_card(hub, head + "x" * pad + "\n\n" + tail + "rollouts during training, not for inference.\n\n" + FILLER)
    assert raw[:CARD_MAX_BYTES].endswith(tail.encode("utf-8"))
    proposal, _, _ = run(env, hub)
    new = note_change(proposal)["fields"]["setup_note"]["new"]
    assert (new["status"], new["package"]) == ("instructions", None)


def test_a_card_longer_than_the_window_is_read_at_its_start_and_a_start_block_there_counts(env):
    hub = Hub()
    F.quiet_world(hub)
    big_card(hub, "# Atlas\n\n" + ATLAS_REQUIRES + "\n" + ATLAS_START + "\n" + FILLER)
    proposal, _, transport = run(env, hub)
    new = note_change(proposal)["fields"]["setup_note"]["new"]
    assert (new["status"], new["package"], new["card_revision"]) == ("instructions", "atlas-inference", F.SHA_A)
    asked = [i for i, (_, url) in enumerate(transport.calls) if url == K.card_url(ROW, F.SHA_A)]
    assert len(asked) == 1 and transport.headers[asked[0]] == {"Range": f"bytes=0-{CARD_MAX_BYTES - 1}"}


@pytest.mark.parametrize("text", [
    "# Atlas\n\n" + FILLER,  # nothing at the start
    "# Atlas\n\n" + FILLER + ATLAS_REQUIRES + "\n" + ATLAS_START,  # the start block only behind the window
], ids=["nothing at the start", "the hit behind the window"])
def test_a_start_of_a_card_without_a_hit_is_not_read_and_the_stored_note_stays(env, text):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    big_card(hub, text)
    proposal, result, _ = run(env, hub)
    assert note_change(proposal) is None and result.exit_code == 1
    assert any(note.startswith(f"{ROW}: model card not read") and "setup note is unchanged" in note for note in proposal["notes"])


def test_a_start_of_a_card_without_a_hit_is_no_statement_that_nothing_was_recognized_without_a_stored_note(env):
    hub = Hub()
    F.quiet_world(hub)
    big_card(hub, "# Atlas\n\n" + FILLER)
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None
    assert any(note.startswith(f"{ROW}: model card not read") for note in proposal["notes"])


def test_the_window_may_end_inside_a_line_and_inside_a_character(env):
    hub = Hub()
    F.quiet_world(hub)
    head = "# Atlas\n\n" + ATLAS_START + "\n"
    pad = CARD_MAX_BYTES - len(head.encode("utf-8")) - 1  # the window ends on the first byte of an `ä`
    raw = big_card(hub, head + "x" * pad + "ä" * 100 + "\n" + FILLER)
    assert raw[CARD_MAX_BYTES - 1:CARD_MAX_BYTES + 1] == "ä".encode("utf-8")
    proposal, _, _ = run(env, hub)
    assert note_change(proposal)["fields"]["setup_note"]["new"]["status"] == "instructions"


def test_a_start_of_a_card_from_another_revision_is_not_taken(env):
    store_note(env, stored_note(package="atlas-inference"))
    hub = Hub()
    F.quiet_world(hub)
    big_card(hub, "# Atlas\n\n" + ATLAS_START + "\n" + FILLER, commit=F.SHA_B)
    proposal, _, _ = run(env, hub)
    assert note_change(proposal) is None and any(note.startswith(f"{ROW}: model card not read") for note in proposal["notes"])
