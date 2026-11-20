# SPDX-License-Identifier: Apache-2.0
"""A reasoning-only completion must not pass the long-document gate."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import qat_battery


@pytest.mark.parametrize(
    ("answer", "finish", "issue"),
    [
        (None, "length", "no final answer"),
        ("<think>unfinished", "length", "reasoning marker in final answer"),
        ("Partial answer", "length", "answer cut off at the output limit"),
        ("Complete answer", "stop", None),
    ],
)
def test_gate_saves_final_answer_separately(
    tmp_path, monkeypatch, answer, finish, issue
):
    reply = {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"reasoning": "working", "content": answer},
            }
        ],
        "usage": {"prompt_tokens": 33, "completion_tokens": 12},
    }

    def respond(request, timeout):
        payload = json.loads(request.data)
        assert payload["model"] == "flint"
        assert payload["thinking_token_budget"] == 8
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(qat_battery.urllib.request, "urlopen", respond)
    args = SimpleNamespace(
        model="flint",
        url="http://localhost/test",
        thinking_token_budget=8,
        api_key="test",
        max_tokens=12,
        out_dir=tmp_path,
        tag="case",
        require_complete=True,
    )
    tokenizer = SimpleNamespace(decode=lambda ids: "document")
    assert qat_battery.run_leg(0, torch.tensor([1]), tokenizer, args) == issue
    assert (tmp_path / "case_leg1.txt").read_text() == (answer or "")
    assert (tmp_path / "case_leg1.reasoning.txt").read_text() == "working"


@pytest.mark.parametrize(
    ("answer", "finish", "valid"),
    [
        ("11", "stop", True),
        ("<think>working</think>11", "stop", False),
        ("There are several steps", "length", False),
    ],
)
def test_short_chat_gate(monkeypatch, answer, finish, valid):
    monkeypatch.syspath_prepend(str(Path(qat_battery.__file__).parent))
    from scripts.probe_bolt_quality import check_row

    choice = {
        "finish_reason": finish,
        "message": {"content": answer},
        "logprobs": {
            "content": [{"logprob": -0.1, "top_logprobs": [{"logprob": -0.1}]}]
        },
    }
    row = {"index": 19, "response": {"choices": [choice]}}
    assert check_row(row, require_complete=True) is valid
