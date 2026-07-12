# Usage: .venv/bin/python convert_bert_onnx.py --language JP

# https://github.com/tuna2134/sbv2-api/blob/main/scripts/convert/convert_deberta.py を参考に実装した
#
# MIT License
# Copyright (c) 2024 tuna2134
# Copyright (c) 2024-2025 tsukumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto
from onnxconverter_common import float16 as float16_converter
from onnxruntime import InferenceSession
from onnxsim import model_info, simplify
from rich import print
from rich.rule import Rule
from rich.style import Style
from torch import nn
from transformers import PreTrainedTokenizerBase

from style_bert_vits2.constants import (
    DEFAULT_BERT_MODEL_PATHS,
    DEFAULT_ONNX_BERT_MODEL_PATHS,
    Languages,
)
from style_bert_vits2.nlp import bert_models


FP32_MAX_DIFF_THRESHOLD = 1e-3
FP32_MEAN_DIFF_THRESHOLD = 1e-4
FP16_MAX_DIFF_THRESHOLD = 2.5e-1
FP16_MEAN_DIFF_THRESHOLD = 2e-3

# 既定の ONNX 推論参照先へ変換結果と一緒に配置する tokenizer / 設定ファイル群
## 変換元 (PyTorch BERT モデルのディレクトリ) に存在するものだけをコピーする
TOKENIZER_FILE_NAMES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "spm.model",
)


def deploy_to_onnx_model_dir(source_dir: Path, deploy_dir: Path) -> list[Path]:
    """
    変換済み ONNX モデルと tokenizer ファイルを、既定の ONNX 推論コードが参照する
    ディレクトリへコピーする。
    既定の ONNX 推論 (style_bert_vits2.nlp.onnx_bert_models) は PyTorch BERT モデルとは
    別の DEFAULT_ONNX_BERT_MODEL_PATHS 配下から model_fp16.onnx と tokenizer をロードするため、
    変換結果をコピーしない限り既定経路からは利用されない。

    ONNX モデル (gitignore 対象) は常に上書きする。一方 tokenizer / 設定ファイルは
    配置先で git tracked な配布資産のため、配置先に無いものだけを補完し、既存ファイルは
    上書きしない (変換スクリプトの実行は git tracked file を変更しない)。

    Args:
        source_dir (Path): 変換結果 (model.onnx / model_fp16.onnx) と tokenizer を含むディレクトリ
        deploy_dir (Path): 既定の ONNX 推論が参照するディレクトリ

    Returns:
        list[Path]: コピーしたファイルのコピー先パスのリスト
    """
    fp16_model_path = source_dir / "model_fp16.onnx"
    if not fp16_model_path.exists():
        raise FileNotFoundError(f"{fp16_model_path} does not exist")

    deploy_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file_name in ("model.onnx", "model_fp16.onnx"):
        source_path = source_dir / file_name
        if not source_path.exists():
            continue
        destination_path = deploy_dir / file_name
        shutil.copy(source_path, destination_path)
        copied.append(destination_path)
    for file_name in TOKENIZER_FILE_NAMES:
        source_path = source_dir / file_name
        destination_path = deploy_dir / file_name
        if not source_path.exists() or destination_path.exists():
            continue
        shutil.copy(source_path, destination_path)
        copied.append(destination_path)
    return copied


def _build_bert_onnx_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(inputs["input_ids"])

    return {
        "input_ids": inputs["input_ids"],
        "token_type_ids": token_type_ids,
        "attention_mask": inputs["attention_mask"],
    }


def _align_float16_cast_attributes(model: onnx.ModelProto) -> int:
    def patch_graph(graph: onnx.GraphProto) -> int:
        value_info_types = {}
        for values in (graph.value_info, graph.input, graph.output):
            for value_info in values:
                if value_info.type.HasField("tensor_type"):
                    value_info_types[value_info.name] = (
                        value_info.type.tensor_type.elem_type
                    )

        patched = 0
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    patched += patch_graph(attr.g)
                for subgraph in attr.graphs:
                    patched += patch_graph(subgraph)

            if node.op_type != "Cast" or len(node.output) == 0:
                continue
            if value_info_types.get(node.output[0]) != TensorProto.FLOAT16:
                continue
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
                    patched += 1

        return patched

    return patch_graph(model.graph)


def validate_model_outputs(
    language: Languages,
    original_model: nn.Module,
    onnx_session: InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    max_diff_threshold: float = FP32_MAX_DIFF_THRESHOLD,
    mean_diff_threshold: float = FP32_MEAN_DIFF_THRESHOLD,
) -> tuple[bool, str]:
    """ONNXモデルの出力を検証"""
    if language == Languages.JP:
        test_texts = [
            "今日はすっごく楽しかったよ！また遊ぼうね！",
            "えー、そんなの嫌だよ。もう二度と行きたくないな…",
            "わーい！プレゼントありがとう！大好き！",
            "あのね、実は昨日泣いちゃったんだ。寂しくて…",
            "もう！怒ったからね！知らないもん！",
            "ごめんなさい…私が悪かったです。許してください。",
            "やったー！テストで100点取れたよ！すっごく嬉しい！",
            "あら、素敵なお洋服ね。とてもお似合いですわ。",
            "うーん、それは難しい選択ですね…よく考えましょう。",
            "おはよう！今日も一日頑張ろうね！元気いっぱいだよ！",
            "こんなに美味しいご飯、初めて食べました！感動です！",
            "ちょっと待って！その話、すっごく気になる！",
            "はぁ…疲れた。今日は本当に大変な一日だったよ。",
            "きゃー！虫！虫がいるよ！誰か助けて！",
            "私ね、将来は宇宙飛行士になりたいの！夢があるでしょ？",
            "あれ？鍵がない…どこに置いたっけ…困ったなぁ…",
            "お誕生日おめでとう！素敵な一年になりますように！",
            "えっと…その…好きです！付き合ってください！",
            "まったく、いつも心配ばかりかけて…でも、ありがとう。",
            "よーし！今日は徹夜で全部やっちゃうぞー！",
            "本日のニュースをお伝えします。",
            "それは、静かな冬の朝のことでした。雪が街を真っ白に染め上げ、人々はまだ深い眠りの中にいました。",
            "ただいまより、会議を始めさせていただきます。",
            "彼女は窓際に立ち、遠く輝く星々を見上げていました。その瞳には、かすかな涙が光っていたのです。",
            "只今の時刻は10時を回りました。",
            "本日のハイライトをお届けいたします。",
            "古びた図書館の奥で、一冊の不思議な本を見つけた少年は、そっとページを開きました。",
            "本製品の特徴について、ご説明いたします。",
            "春風が桜の花びらを舞わせる中、彼は十年ぶりに故郷の駅に降り立ちました。",
            "続いて、天気予報です。",
            "お客様へのご案内を申し上げます。",
            "深い森の中、こだまする鳥のさえずりと、せせらぎの音だけが時を刻んでいました。",
            "月明かりに照らされた海は、まるで無数の宝石を散りばめたように輝いていました。",
            "このたびの新商品発表会にようこそ。",
            "古城の階段を上るたび、彼女の心臓は激しく鼓動を打ちました。この先で何が待ち受けているのか…",
            "本日のメニューをご紹介いたします。",
            "時計の針が真夜中を指す瞬間、不思議な出来事の幕が開けたのです。",
            "次の停車駅は、東京駅です。",
            "霧深い港町で、一通の差出人不明の手紙が彼を待っていました。",
            "幼い頃に聞いた祖母の物語は、今でも鮮明に心に刻まれています。",
        ]
    elif language == Languages.EN:
        test_texts = [
            "Today was so much fun! Let's play again!",
            "Ugh, I hate that. I never want to go there again...",
            "Yay! Thank you for the present! I love it!",
            "You know, I actually cried yesterday. I was feeling lonely...",
            "That's it! I'm angry! I don't care anymore!",
            "I'm sorry... It was my fault. Please forgive me.",
            "I did it! I got 100 on the test! I'm so happy!",
            "My, what a lovely dress. It suits you perfectly.",
            "Hmm, that's a difficult choice... Let's think about it carefully.",
            "Good morning! Let's do our best today! I'm full of energy!",
            "This is the most delicious food I've ever had! I'm moved!",
            "Wait a minute! That story sounds really interesting!",
            "Sigh... I'm tired. Today was really tough.",
            "Eek! A bug! A bug! Someone help!",
            "You know what? I want to become an astronaut! Isn't that a great dream?",
            "Huh? Where are my keys... Where did I put them... This is troubling...",
            "Happy birthday! May you have a wonderful year ahead!",
            "Um... well... I like you! Please go out with me!",
            "Geez, you always make me worry... but thank you.",
            "Alright! I'm going to pull an all-nighter and finish everything today!",
            "Now for today's news.",
            "It was a quiet winter morning. Snow had painted the town white, and people were still deep in slumber.",
            "Let us now begin the meeting.",
            "She stood by the window, gazing at the distant stars. Tears glistened faintly in her eyes.",
            "The time is now just past 10 o'clock.",
            "Here are today's highlights.",
            "In the depths of an old library, a boy found a mysterious book and gently opened its pages.",
            "Let me explain the features of this product.",
            "As spring winds scattered cherry blossoms, he stepped onto his hometown station platform for the first time in ten years.",
            "And now, the weather forecast.",
            "An announcement for our customers.",
            "Deep in the forest, only the echoing birdsong and the sound of flowing water marked the passage of time.",
            "The moonlit sea sparkled like countless scattered jewels.",
            "Welcome to today's new product announcement.",
            "With each step up the castle stairs, her heart beat faster, wondering what awaited ahead...",
            "Let me introduce today's menu.",
            "As the clock struck midnight, the curtain rose on a strange occurrence.",
            "The next stop is Tokyo Station.",
            "In the foggy port town, a letter with no sender awaited him.",
            "The story my grandmother told me in my childhood remains vivid in my heart.",
        ]
    elif language == Languages.ZH:
        test_texts = [
            "今天真是太开心了！下次再一起玩吧！",
            "唉，我讨厌那样。我再也不想去那里了...",
            "耶！谢谢你的礼物！我好喜欢！",
            "其实呢，我昨天哭了。因为感到很寂寞...",
            "够了！我生气了！我不管了！",
            "对不起...都是我的错。请原谅我。",
            "太棒了！考试得了100分！我好高兴！",
            "哎呀，多漂亮的衣服啊。真适合你。",
            "嗯，这是个难决定...让我们好好考虑一下。",
            "早安！今天也要加油哦！我充满干劲！",
            "这是我吃过最好吃的饭！太感动了！",
            "等一下！那个故事听起来很有趣！",
            "唉...好累啊。今天真是辛苦的一天。",
            "啊！虫子！虫子！谁来帮帮我！",
            "你知道吗？我想成为宇航员！这是个好梦想吧？",
            "咦？钥匙呢...放在哪里了...真是麻烦...",
            "生日快乐！祝你度过美好的一年！",
            "那个...就是...我喜欢你！请和我交往！",
            "真是的，总是让人担心...不过，谢谢你。",
            "好！今天我要熬夜把所有事情都做完！",
            "现在播报今日新闻。",
            "那是个安静的冬日早晨。白雪覆盖了整个城镇，人们还在沉睡中。",
            "现在开始会议。",
            "她站在窗边，仰望着远处的星星。她的眼中闪烁着微弱的泪光。",
            "现在时间刚过十点。",
            "为您播报今日要闻。",
            "在古老图书馆的深处，一个男孩发现了一本神秘的书，轻轻地翻开了书页。",
            "让我为您介绍本产品的特点。",
            "春风吹散樱花花瓣时，他时隔十年重返故乡车站。",
            "接下来是天气预报。",
            "现在为顾客播报通知。",
            "在深邃的森林中，只有鸟鸣的回声和溪水的声音在记录着时间的流逝。",
            "月光照耀下的大海，闪烁得像撒满了无数宝石。",
            "欢迎参加今天的新产品发布会。",
            "每上一级城堡的台阶，她的心跳就加快一分，不知前方等待着什么...",
            "让我为您介绍今日菜单。",
            "当时钟指向午夜时分，一个奇异的事件拉开了序幕。",
            "下一站是东京站。",
            "在雾气弥漫的港口小镇，一封没有署名的信在等待着他。",
            "童年时奶奶讲的故事，至今仍清晰地铭刻在我的心中。",
        ]

    max_diff = 0
    mean_diff = 0

    # セッションの入力名を取得
    input_names = [input.name for input in onnx_session.get_inputs()]

    original_model.eval()
    with torch.no_grad():
        for text in test_texts:
            # PyTorch
            inputs = _build_bert_onnx_inputs(tokenizer(text, return_tensors="pt"))
            torch_output = original_model(
                inputs["input_ids"],
                inputs["token_type_ids"],
                inputs["attention_mask"],
            ).numpy()

            # ONNX
            onnx_inputs = {}
            if "input_ids" in input_names:
                onnx_inputs["input_ids"] = inputs["input_ids"].numpy().astype(np.int64)  # type: ignore
            if "token_type_ids" in input_names:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].numpy().astype(np.int64)  # type: ignore
            if "attention_mask" in input_names:
                onnx_inputs["attention_mask"] = inputs["attention_mask"].numpy().astype(np.int64)  # type: ignore

            onnx_output = onnx_session.run(None, onnx_inputs)[0]

            # 差分を計算
            diff = np.abs(torch_output - onnx_output)
            max_diff = max(max_diff, np.max(diff))
            mean_diff = max(mean_diff, np.mean(diff))

    is_valid = max_diff < max_diff_threshold and mean_diff < mean_diff_threshold
    message = (
        f"Validation {'passed' if is_valid else 'failed'}\n"
        f"Max difference: {max_diff:.6f} (threshold: {max_diff_threshold})\n"
        f"Mean difference: {mean_diff:.6f} (threshold: {mean_diff_threshold})"
    )
    return is_valid, message


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument(
        "--language",
        default=Languages.JP,
        help="Language of the BERT model to be converted",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip copying the converted models and tokenizer files to the default ONNX BERT model directory",
    )
    args = parser.parse_args()

    # モデルの入出力先ファイルパスを取得
    language = Languages(args.language)
    pretrained_model_name_or_path = DEFAULT_BERT_MODEL_PATHS[language]
    onnx_temp_model_path = Path(pretrained_model_name_or_path) / "model_temp.onnx"
    onnx_fp32_model_path = Path(pretrained_model_name_or_path) / "model.onnx"
    onnx_fp16_model_path = Path(pretrained_model_name_or_path) / "model_fp16.onnx"

    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Language:[/bold cyan] {language.name}")
    print(f"[bold cyan]Pretrained model:[/bold cyan] {pretrained_model_name_or_path}")
    print(Rule(characters="=", style=Style(color="blue")))

    class ONNXBert(nn.Module):
        def __init__(self):
            super(ONNXBert, self).__init__()
            self.model = bert_models.load_model(language)

        def forward(self, input_ids, token_type_ids, attention_mask):
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
            return res

    # 再度 Fast Tokenizer でロード
    tokenizer = bert_models.load_tokenizer(language, str(pretrained_model_name_or_path))

    # ONNX 変換用の BERT モデルをロード
    model = ONNXBert()
    inputs = _build_bert_onnx_inputs(
        tokenizer("今日はいい天気ですね", return_tensors="pt")
    )

    # モデルを ONNX に変換
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Exporting ONNX model...[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    export_start_time = time.time()
    torch.onnx.export(
        model=model,
        args=(
            inputs["input_ids"],
            inputs["token_type_ids"],
            inputs["attention_mask"],
        ),
        f=str(onnx_temp_model_path),
        verbose=False,
        input_names=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
        ],
        output_names=["output"],
        dynamo=False,
        opset_version=20,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print(
        f"[bold green]ONNX model exported ({time.time() - export_start_time:.2f}s)[/bold green]"
    )

    # ONNX モデルを最適化
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Optimizing ONNX model...[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    optimize_start_time = time.time()
    onnx_model = onnx.load(onnx_temp_model_path)
    simplified_onnx_model, check = simplify(onnx_model)
    onnx.save(simplified_onnx_model, onnx_fp32_model_path)
    print(
        f"[bold green]ONNX model optimized ({time.time() - optimize_start_time:.2f}s)[/bold green]"
    )

    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Optimized ONNX model info:[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    model_info.print_simplifying_info(onnx_model, simplified_onnx_model)

    # FP32 モデルの検証
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Validating FP32 model...[/bold cyan]")
    session = InferenceSession(
        str(onnx_fp32_model_path),
        providers=["CPUExecutionProvider"],
    )
    is_valid, message = validate_model_outputs(language, model, session, tokenizer)
    color = "green" if is_valid else "red"
    print(f"[bold {color}]{message}[/bold {color}]")
    if not is_valid:
        raise RuntimeError(message)

    # FP16 への変換
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Converting to FP16...[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    fp16_start_time = time.time()
    fp16_model = float16_converter.convert_float_to_float16(
        simplified_onnx_model,
        keep_io_types=True,  # 入出力は float32 のまま
        disable_shape_infer=True,
    )
    _align_float16_cast_attributes(fp16_model)
    onnx.save(fp16_model, onnx_fp16_model_path)
    print(
        f"[bold green]FP16 conversion completed ({time.time() - fp16_start_time:.2f}s)[/bold green]"
    )

    # FP16 モデルの検証
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Validating FP16 model...[/bold cyan]")
    session = InferenceSession(
        str(onnx_fp16_model_path),
        providers=["CPUExecutionProvider"],
    )
    is_valid, message = validate_model_outputs(
        language,
        model,
        session,
        tokenizer,
        max_diff_threshold=FP16_MAX_DIFF_THRESHOLD,
        mean_diff_threshold=FP16_MEAN_DIFF_THRESHOLD,
    )
    color = "green" if is_valid else "red"
    print(f"[bold {color}]{message}[/bold {color}]")
    if not is_valid:
        raise RuntimeError(message)

    # 検証済みの変換結果を、既定の ONNX 推論が参照するディレクトリへ配置
    ## --no-deploy 指定時はスキップし、変換元ディレクトリへの出力だけを行う
    if not args.no_deploy:
        print(Rule(characters="=", style=Style(color="blue")))
        print("[bold cyan]Deploying to the default ONNX BERT model directory...[/bold cyan]")
        print(Rule(characters="=", style=Style(color="blue")))
        deploy_dir = DEFAULT_ONNX_BERT_MODEL_PATHS[language]
        copied_paths = deploy_to_onnx_model_dir(
            Path(pretrained_model_name_or_path), deploy_dir
        )
        print(
            f"[bold green]Deployed {len(copied_paths)} files to {deploy_dir}[/bold green]"
        )

    # サイズ情報の表示
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Model size information:[/bold cyan]")
    original_size = onnx_temp_model_path.stat().st_size / 1000 / 1000
    fp32_size = onnx_fp32_model_path.stat().st_size / 1000 / 1000
    fp16_size = onnx_fp16_model_path.stat().st_size / 1000 / 1000
    print(f"Original: {original_size:.2f}MB")
    print(f"Optimized (FP32): {fp32_size:.2f}MB")
    print(f"Optimized (FP16): {fp16_size:.2f}MB")
    print(f"Size reduction: {(1 - fp16_size/original_size) * 100:.1f}%")

    # 一時ファイルの削除
    onnx_temp_model_path.unlink()

    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold green]Total time: {time.time() - start_time:.2f}s[/bold green]")
    print(Rule(characters="=", style=Style(color="blue")))
