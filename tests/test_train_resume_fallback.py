import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAINING_SCRIPT = Path(__file__).parents[1] / "train_ms_jp_extra.py"


def _optional_resume_try(checkpoint_pattern: str) -> ast.Try:
    tree = ast.parse(TRAINING_SCRIPT.read_text(encoding="utf-8"))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Constant) and child.value == checkpoint_pattern
            for child in ast.walk(node)
        )
    ]
    assert len(candidates) == 1, (
        f"expected one optional resume block for {checkpoint_pattern}, "
        f"found {len(candidates)}"
    )
    return candidates[0]


@pytest.mark.parametrize(
    ("checkpoint_pattern", "optimizer_name", "discriminator_name"),
    [
        ("DUR_*.pth", "optim_dur_disc", "net_dur_disc"),
        ("WD_*.pth", "optim_wd", "net_wd"),
    ],
)
def test_optional_discriminator_resume_failure_uses_configured_initial_lr(
    checkpoint_pattern: str,
    optimizer_name: str,
    discriminator_name: str,
):
    learning_rate = 2e-4
    optimizer = SimpleNamespace(param_groups=[{}])

    def load_checkpoint(*args, **kwargs):
        raise RuntimeError("broken optional checkpoint")

    checkpoints = SimpleNamespace(
        get_latest_checkpoint_path=lambda model_dir, pattern: model_dir / pattern,
        load_checkpoint=load_checkpoint,
    )
    namespace = {
        "hps": SimpleNamespace(
            train=SimpleNamespace(
                learning_rate=learning_rate,
                skip_optimizer=False,
            )
        ),
        "logger": SimpleNamespace(info=lambda message: None),
        "model_dir": Path("models"),
        "utils": SimpleNamespace(checkpoints=checkpoints),
        optimizer_name: optimizer,
        discriminator_name: object(),
    }
    resume_try = _optional_resume_try(checkpoint_pattern)
    isolated_block = ast.fix_missing_locations(
        ast.Module(body=[resume_try], type_ignores=[])
    )

    exec(compile(isolated_block, TRAINING_SCRIPT, "exec"), namespace)

    assert optimizer.param_groups[0]["initial_lr"] == learning_rate
