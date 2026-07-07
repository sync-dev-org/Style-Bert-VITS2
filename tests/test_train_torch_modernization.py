import ast
import warnings
from pathlib import Path

import torch

from mel_processing import mel_spectrogram_torch, spectrogram_torch
from style_bert_vits2.models import models, models_jp_extra, modules
from style_bert_vits2.models.utils.checkpoints import load_checkpoint


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "train_torch_modernization"
REPO_ROOT = Path(__file__).parents[1]


def _small_generator() -> models.Generator:
    return models.Generator(
        initial_channel=4,
        resblock_str="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_rates=[2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4],
        gin_channels=0,
    )


def test_mel_processing_matches_pre_migration_reference():
    fixture = torch.load(FIXTURE_DIR / "mel_processing_reference.pt", weights_only=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spectrogram = spectrogram_torch(
            fixture["input"], **fixture["spectrogram_params"]
        )
        mel = mel_spectrogram_torch(fixture["input"], **fixture["mel_params"])

    torch.testing.assert_close(
        spectrogram, fixture["expected_spectrogram"], atol=1e-6, rtol=0
    )
    torch.testing.assert_close(mel, fixture["expected_mel"], atol=1e-6, rtol=0)
    scoped_warnings = [
        warning
        for warning in caught
        if isinstance(warning.message, (DeprecationWarning, FutureWarning))
        and any(token in str(warning.message) for token in ("return_complex", "stft"))
    ]
    assert scoped_warnings == []


def test_legacy_weight_norm_generator_checkpoint_loads_into_new_keys():
    checkpoint = torch.load(
        FIXTURE_DIR / "legacy_weight_norm_generator_checkpoint.pt", weights_only=True
    )
    model = _small_generator()

    model, _optimizer, learning_rate, iteration = load_checkpoint(
        FIXTURE_DIR / "legacy_weight_norm_generator_checkpoint.pt",
        model,
        optimizer=None,
        skip_optimizer=True,
        device="cpu",
    )

    state_dict = model.state_dict()
    assert iteration == 7
    assert learning_rate == 1e-4
    assert "ups.0.parametrizations.weight.original0" in state_dict
    assert "ups.0.parametrizations.weight.original1" in state_dict
    assert not any(key.endswith(("weight_g", "weight_v")) for key in state_dict)
    torch.testing.assert_close(
        state_dict["ups.0.parametrizations.weight.original0"],
        checkpoint["model"]["ups.0.weight_g"],
    )
    torch.testing.assert_close(
        state_dict["ups.0.parametrizations.weight.original1"],
        checkpoint["model"]["ups.0.weight_v"],
    )

    x = torch.ones(1, 4, 3)
    y = model(x)
    assert y.shape == (1, 1, 6)
    assert torch.isfinite(y).all()


def test_model_construction_emits_no_torch_modernization_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modules.ResBlock1(4)
        modules.ResBlock2(4)
        modules.WN(4, 3, 1, 1, gin_channels=4)
        _small_generator()
        models_jp_extra.Generator(
            initial_channel=4,
            resblock_str="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_rates=[2],
            upsample_initial_channel=8,
            upsample_kernel_sizes=[4],
            gin_channels=0,
        )

    scoped_warnings = [
        warning
        for warning in caught
        if isinstance(warning.message, (DeprecationWarning, FutureWarning))
        and any(
            token in str(warning.message)
            for token in ("weight_norm", "return_complex", "sdp_kernel")
        )
    ]
    assert scoped_warnings == []


def test_training_scripts_do_not_call_deprecated_sdp_kernel():
    offenders = []
    for relative_path in (Path("train_ms.py"), Path("train_ms_jp_extra.py")):
        tree = ast.parse((REPO_ROOT / relative_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "sdp_kernel":
                    offenders.append(f"{relative_path}:{node.lineno}")

    assert offenders == []


def test_all_torch_load_calls_spell_out_weights_only():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part.startswith(".") for part in parts) or "__pycache__" in parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and not any(keyword.arg == "weights_only" for keyword in node.keywords)
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []
