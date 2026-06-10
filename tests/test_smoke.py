"""Smoke tests — fast, offline checks that the core wiring is intact.

No network calls. Validates the label contract, every prompt style builds, and
the JSON parser is robust. Run with:  pytest -q
"""
import pytest

from config import LABELS, LABEL_DESCRIPTIONS
import prompts


def test_label_contract():
    assert len(LABELS) == 6
    # every label has a human-readable description and vice versa
    assert set(LABEL_DESCRIPTIONS) == set(LABELS)


@pytest.mark.parametrize("style", prompts.STYLES)
def test_build_prompt_each_style(style):
    system, user = prompts.build_prompt(style, "hello world")
    assert isinstance(user, str) and "hello world" in user
    assert system is None or isinstance(system, str)


def test_build_prompt_rejects_unknown_style():
    with pytest.raises(ValueError):
        prompts.build_prompt("does_not_exist", "x")


def test_style_to_exp_mapping_complete():
    assert set(prompts.STYLE_TO_EXP) == set(prompts.STYLES)


# parse_json lives in llm_client, which constructs API keys at import time.
# Skip these gracefully if no keys are configured (e.g. in a bare CI env).
try:
    from llm_client import parse_json
    _HAS_CLIENT = True
except Exception:
    _HAS_CLIENT = False

skip_no_client = pytest.mark.skipif(not _HAS_CLIENT, reason="llm_client unavailable (no API keys configured)")


@skip_no_client
def test_parse_json_clean():
    assert parse_json('{"label": "age"}')["label"] == "age"


@skip_no_client
def test_parse_json_repairs_trailing_comma():
    assert parse_json('{"label": "age",}')["label"] == "age"


@skip_no_client
def test_parse_json_extracts_from_prose():
    assert parse_json('Sure! {"label": "gender"} done')["label"] == "gender"


@skip_no_client
def test_parse_json_flags_garbage():
    assert parse_json("not json at all")["_parse_error"] is True
