from __future__ import annotations

import pytest


pytest.importorskip("torch", reason="requires the torch optional dependency")

from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)


# 構築可否の検証だけが目的のため、各次元は最小限に抑える
_SMALL_MODEL_KWARGS = dict(
    n_vocab=64,
    spec_channels=128,
    segment_size=32,
    inter_channels=32,
    hidden_channels=32,
    filter_channels=32,
    n_heads=2,
    n_layers=3,
    kernel_size=3,
    p_dropout=0.1,
    resblock="2",
    resblock_kernel_sizes=[3],
    resblock_dilation_sizes=[[1, 3]],
    upsample_rates=[4, 4],
    upsample_initial_channel=16,
    upsample_kernel_sizes=[8, 8],
    n_speakers=2,
    use_transformer_flow=False,
)


@pytest.mark.parametrize(
    "synthesizer_class", [SynthesizerTrn, SynthesizerTrnJPExtra]
)
def test_construction_without_spk_conditioned_encoder(synthesizer_class):
    model = synthesizer_class(
        **_SMALL_MODEL_KWARGS,
        gin_channels=32,
        use_spk_conditioned_encoder=False,
    )

    assert model.enc_p.gin_channels == 0


@pytest.mark.parametrize(
    "synthesizer_class", [SynthesizerTrn, SynthesizerTrnJPExtra]
)
def test_construction_with_zero_gin_channels(synthesizer_class):
    model = synthesizer_class(**_SMALL_MODEL_KWARGS, gin_channels=0)

    assert model.enc_p.gin_channels == 0
