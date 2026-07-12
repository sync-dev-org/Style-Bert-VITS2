from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from preprocess_all import main


CLI_ARGS = ["--model_name", "example"]
PIPELINE_KWARGS = {
    "model_name": "example",
    "batch_size": 2,
    "epochs": 100,
    "save_every_steps": 1000,
    "num_processes": 1,
    "normalize": False,
    "trim": False,
    "freeze_EN_bert": False,
    "freeze_JP_bert": False,
    "freeze_ZH_bert": False,
    "freeze_style": False,
    "freeze_decoder": False,
    "use_jp_extra": False,
    "val_per_lang": 0,
    "log_interval": 200,
    "yomi_error": "raise",
}


def test_main_returns_nonzero_when_preprocess_fails(capsys):
    def fail(**_kwargs: Any) -> tuple[bool, str]:
        return False, "Step 5 failed"

    exit_code = main(CLI_ARGS, preprocess=fail)

    assert exit_code == 1
    assert "Step 5 failed" in capsys.readouterr().err


def test_main_returns_zero_when_preprocess_succeeds():
    def succeed(**_kwargs: Any) -> tuple[bool, str]:
        return True, "finished"

    assert main(CLI_ARGS, preprocess=succeed) == 0


@pytest.mark.parametrize(
    "failed_step",
    ["initialize", "resample", "preprocess_text", "bert_gen", "style_gen"],
)
def test_pipeline_stops_and_reports_every_failed_step(monkeypatch, failed_step: str):
    pytest.importorskip("gradio", reason="requires the webui dependency group")
    from gradio_tabs import train

    step_names = ["initialize", "resample", "preprocess_text", "bert_gen", "style_gen"]
    calls: list[str] = []

    def step(name: str) -> Callable[..., tuple[bool, str]]:
        def run(**_kwargs: Any) -> tuple[bool, str]:
            calls.append(name)
            if name == failed_step:
                return False, f"{name} failed"
            return True, f"{name} succeeded"

        return run

    for name in step_names:
        monkeypatch.setattr(train, name, step(name))

    success, message = train.preprocess_all(**PIPELINE_KWARGS)

    failed_index = step_names.index(failed_step)
    assert success is False
    assert message == f"{failed_step} failed"
    assert calls == step_names[: failed_index + 1]
