import ast
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).parents[1]
TRAINING_SCRIPTS = (Path("train_ms.py"), Path("train_ms_jp_extra.py"))


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def _is_final_epoch_compare(node: ast.AST) -> bool:
    # `epoch == hps.train.epochs`
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "epoch"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and ast.unparse(node.comparators[0]) == "hps.train.epochs"
    )


def _is_rank_zero_compare(node: ast.AST) -> bool:
    # `rank == 0`
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "rank"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
    )


def _guarded_by_rank_zero(if_node: ast.If) -> bool:
    node: ast.AST = if_node
    while node is not None:
        if isinstance(node, ast.If) and any(
            _is_rank_zero_compare(child) for child in ast.walk(node.test)
        ):
            return True
        node = getattr(node, "_parent", None)
    return False


def test_final_epoch_save_is_guarded_by_rank_zero():
    for relative_path in TRAINING_SCRIPTS:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        _attach_parents(tree)
        final_epoch_ifs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(_is_final_epoch_compare(child) for child in ast.walk(node.test))
        ]
        assert final_epoch_ifs, f"{relative_path}: final-epoch save block not found"
        offenders = [
            f"{relative_path}:{if_node.lineno}"
            for if_node in final_epoch_ifs
            if not _guarded_by_rank_zero(if_node)
        ]
        assert offenders == [], (
            "final-epoch checkpoint/safetensors saving must run only on rank 0 "
            f"(unguarded blocks: {offenders})"
        )


def test_training_scripts_have_no_assets_root_cli_argument():
    for relative_path in TRAINING_SCRIPTS:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        offenders = [
            f"{relative_path}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(arg, ast.Constant) and arg.value == "--assets_root"
                for arg in node.args
            )
        ]
        assert offenders == [], (
            "the assets root is configured solely by configs/paths.yml; "
            "training scripts must not expose an --assets_root CLI argument "
            f"({offenders})"
        )


def test_hyper_parameters_train_has_no_fp16_run_field():
    from style_bert_vits2.models.hyper_parameters import HyperParametersTrain

    assert "fp16_run" not in HyperParametersTrain.model_fields


def test_hyper_parameters_load_ignores_legacy_fp16_run(tmp_path):
    from style_bert_vits2.models.hyper_parameters import HyperParameters

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"train": {"fp16_run": True, "bf16_run": True}}),
        encoding="utf-8",
    )

    hps = HyperParameters.load_from_json(config_path)

    assert hps.train.bf16_run is True
    assert not hasattr(hps.train, "fp16_run")


def test_config_templates_have_no_fp16_run():
    for name in ("config.json", "config_jp_extra.json"):
        template = json.loads(
            (REPO_ROOT / "configs" / name).read_text(encoding="utf-8")
        )
        assert "fp16_run" not in template["train"], f"configs/{name}"


def test_bert_gen_config_ignores_legacy_num_processes():
    pytest.importorskip("torch", reason="config.py imports torch")
    from config import Bert_gen_config

    bert_gen_config = Bert_gen_config.from_dict(
        Path("Data/Dummy"),
        {
            "config_path": "config.json",
            "num_processes": 4,
            "device": "cpu",
            "use_multi_device": False,
        },
    )

    assert bert_gen_config.config_path == Path("Data/Dummy/config.json")
    assert not hasattr(bert_gen_config, "num_processes")


def test_default_config_bert_gen_has_no_num_processes():
    default_config = yaml.safe_load(
        (REPO_ROOT / "default_config.yml").read_text(encoding="utf-8")
    )
    assert "num_processes" not in default_config["bert_gen"]


def test_bert_gen_executor_is_pinned_to_single_worker():
    tree = ast.parse((REPO_ROOT / "bert_gen.py").read_text(encoding="utf-8"))
    executor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "ThreadPoolExecutor")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "ThreadPoolExecutor"
            )
        )
    ]
    assert executor_calls, "bert_gen.py: ThreadPoolExecutor call not found"
    for call in executor_calls:
        max_workers = next(
            (kw.value for kw in call.keywords if kw.arg == "max_workers"), None
        )
        assert isinstance(max_workers, ast.Constant) and max_workers.value == 1, (
            "bert_gen.py: the executor must stay single-worker because the "
            "pyopenjtalk worker client shares one unsynchronized socket"
        )
