import ast
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).parents[1]


def _parse(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text())


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent is None:
            return node.attr
        return f"{parent}.{node.attr}"
    return None


def _node_mentions(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in names for child in ast.walk(node)
    )


def _constant_assignments(relative_path: str) -> dict[str, object]:
    constants = {}
    for node in _parse(relative_path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.isupper():
            constants[target.id] = ast.literal_eval(node.value)
    return constants


def _load_function_from_script(relative_path: str, function_name: str):
    tree = _parse(relative_path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"onnx": onnx, "TensorProto": TensorProto, "torch": torch}
            exec(compile(module, str(REPO_ROOT / relative_path), "exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"{function_name} is not defined in {relative_path}")


def test_all_torch_onnx_exports_pin_torchscript_exporter_and_opset():
    offenders = []
    for relative_path in ("convert_onnx.py", "convert_bert_onnx.py"):
        tree = _parse(relative_path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _call_name(node.func) == "torch.onnx.export"
            ):
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                dynamo = keywords.get("dynamo")
                opset_version = keywords.get("opset_version")
                if not (isinstance(dynamo, ast.Constant) and dynamo.value is False):
                    offenders.append(f"{relative_path}:{node.lineno} missing dynamo=False")
                if not (
                    isinstance(opset_version, ast.Constant)
                    and opset_version.value == 20
                ):
                    offenders.append(
                        f"{relative_path}:{node.lineno} missing opset_version=20"
                    )

    assert offenders == []


def test_aivm_generation_references_are_removed_from_onnx_conversion_paths():
    offenders = []
    for relative_path in ("convert_onnx.py", "convert_bert_onnx.py"):
        text = (REPO_ROOT / relative_path).read_text()
        for token in ("aivmlib", "--aivm", "--aivmx", "AIVM", "AIVMX"):
            if token in text:
                offenders.append(f"{relative_path} contains {token}")

    assert offenders == []


def test_ui_and_readme_do_not_route_onnx_users_to_aivis_or_aivm_tools():
    offenders = []
    for relative_path in ("gradio_tabs/convert_onnx.py", "README.md"):
        text = (REPO_ROOT / relative_path).read_text()
        for token in (
            "AIVM Generator",
            "AivisSpeech",
            "Aivis Speech",
            "AIVM形式",
            "AIVMX形式",
        ):
            if token in text:
                offenders.append(f"{relative_path} contains {token}")

    assert offenders == []


def test_convert_bert_onnx_does_not_save_tokenizers_into_default_model_tree():
    tree = _parse("convert_bert_onnx.py")
    offenders = []
    default_model_path_names = {
        "DEFAULT_BERT_MODEL_PATHS",
        "pretrained_model_name_or_path",
        "tokenizer_json_path",
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"save", "save_pretrained"}
        ):
            if any(_node_mentions(arg, default_model_path_names) for arg in node.args):
                offenders.append(f"convert_bert_onnx.py:{node.lineno}")
            if any(
                keyword.value is not None
                and _node_mentions(keyword.value, default_model_path_names)
                for keyword in node.keywords
            ):
                offenders.append(f"convert_bert_onnx.py:{node.lineno}")

    assert offenders == []


def test_bert_onnx_input_builder_synthesizes_missing_token_type_ids():
    build_inputs = _load_function_from_script(
        "convert_bert_onnx.py", "_build_bert_onnx_inputs"
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.int64)

    inputs = build_inputs(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    assert inputs["input_ids"] is input_ids
    assert inputs["attention_mask"] is attention_mask
    torch.testing.assert_close(inputs["token_type_ids"], torch.zeros_like(input_ids))
    assert inputs["token_type_ids"].dtype == input_ids.dtype
    assert inputs["token_type_ids"].device == input_ids.device


def test_bert_onnx_input_builder_preserves_token_type_ids_when_available():
    build_inputs = _load_function_from_script(
        "convert_bert_onnx.py", "_build_bert_onnx_inputs"
    )
    token_type_ids = torch.tensor([[0, 1, 0]], dtype=torch.int64)

    inputs = build_inputs(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "token_type_ids": token_type_ids,
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }
    )

    assert inputs["token_type_ids"] is token_type_ids


def test_float16_cast_alignment_updates_casts_that_feed_fp16_values():
    align_casts = _load_function_from_script(
        "convert_bert_onnx.py", "_align_float16_cast_attributes"
    )
    cast_node = helper.make_node(
        "Cast",
        inputs=["input_ids"],
        outputs=["cast_output"],
        to=TensorProto.FLOAT,
    )
    graph = helper.make_graph(
        [cast_node],
        "cast-test",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 3])],
        [helper.make_tensor_value_info("cast_output", TensorProto.FLOAT16, [1, 3])],
        value_info=[
            helper.make_tensor_value_info("cast_output", TensorProto.FLOAT16, [1, 3])
        ],
    )
    model = helper.make_model(graph)

    patched = align_casts(model)

    assert patched == 1
    cast_to = next(attr.i for attr in model.graph.node[0].attribute if attr.name == "to")
    assert cast_to == TensorProto.FLOAT16


def test_bert_fp16_validation_thresholds_are_explicit_for_current_ort_export():
    constants = _constant_assignments("convert_bert_onnx.py")

    assert constants["FP16_MAX_DIFF_THRESHOLD"] == 2.5e-1
    assert constants["FP16_MEAN_DIFF_THRESHOLD"] == 2e-3
