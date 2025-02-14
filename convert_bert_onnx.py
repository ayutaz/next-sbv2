# Usage: .venv/bin/python convert_bert_onnx.py --language JP
# ref: https://github.com/tuna2134/sbv2-api/blob/main/convert/convert_deberta.py

import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxconverter_common import float16 as float16_converter
from onnxruntime import InferenceSession
from onnxsim import model_info, simplify
from rich import print
from rich.rule import Rule
from rich.style import Style
from torch import nn
from transformers import AutoTokenizer, DebertaV2Tokenizer, PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import BertConverter, convert_slow_tokenizer

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models


def validate_model_outputs(
    language: Languages,
    original_model: nn.Module,
    onnx_session: InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    max_diff_threshold: float = 1e-3,
    mean_diff_threshold: float = 1e-4,
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

    max_diff = 0
    mean_diff = 0

    # セッションの入力名を取得
    input_names = [input.name for input in onnx_session.get_inputs()]

    original_model.eval()
    with torch.no_grad():
        for text in test_texts:
            # PyTorch
            inputs = tokenizer(text, return_tensors="pt")
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
    args = parser.parse_args()

    # モデルの入出力先ファイルパスを取得
    language = Languages(args.language)
    pretrained_model_name_or_path = DEFAULT_BERT_MODEL_PATHS[language]
    onnx_temp_model_path = Path(pretrained_model_name_or_path) / f"model_temp.onnx"
    onnx_fp32_model_path = Path(pretrained_model_name_or_path) / f"model.onnx"
    onnx_fp16_model_path = Path(pretrained_model_name_or_path) / f"model_fp16.onnx"
    tokenizer_json_path = Path(pretrained_model_name_or_path) / "tokenizer.json"

    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Language:[/bold cyan] {language.name}")
    print(f"[bold cyan]Pretrained model:[/bold cyan] {pretrained_model_name_or_path}")
    print(Rule(characters="=", style=Style(color="blue")))

    # トークナイザーを Fast Tokenizer 用形式に変換して保存
    if language == Languages.JP:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=False,  # 明示的に Slow Tokenizer を使う
        )
        BertConverter(tokenizer).converted().save(str(tokenizer_json_path))
    else:
        assert False, "Invalid language"
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold green]Tokenizer JSON saved to {tokenizer_json_path}[/bold green]")
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
    tokenizer = bert_models.load_tokenizer(language)

    # ONNX 変換用の BERT モデルをロード
    model = ONNXBert()
    inputs = tokenizer("今日はいい天気ですね", return_tensors="pt")

    # モデルを ONNX に変換
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Exporting ONNX model...[/bold cyan]")
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
    print(f"[bold cyan]Optimizing ONNX model...[/bold cyan]")
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
    print(f"[bold cyan]Validating FP32 model...[/bold cyan]")
    session = InferenceSession(
        str(onnx_fp32_model_path),
        providers=["CPUExecutionProvider"],
    )
    is_valid, message = validate_model_outputs(language, model, session, tokenizer)
    color = "green" if is_valid else "red"
    print(f"[bold {color}]{message}[/bold {color}]")

    if is_valid:
        # FP16 への変換
        print(Rule(characters="=", style=Style(color="blue")))
        print(f"[bold cyan]Converting to FP16...[/bold cyan]")
        print(Rule(characters="=", style=Style(color="blue")))
        fp16_start_time = time.time()
        fp16_model = float16_converter.convert_float_to_float16(
            simplified_onnx_model,
            keep_io_types=True,  # 入出力は float32 のまま
            disable_shape_infer=True,
        )
        onnx.save(fp16_model, onnx_fp16_model_path)
        print(
            f"[bold green]FP16 conversion completed ({time.time() - fp16_start_time:.2f}s)[/bold green]"
        )

        # FP16 モデルの検証
        print(Rule(characters="=", style=Style(color="blue")))
        print(f"[bold cyan]Validating FP16 model...[/bold cyan]")
        session = InferenceSession(
            str(onnx_fp16_model_path),
            providers=["CPUExecutionProvider"],
        )
        is_valid, message = validate_model_outputs(
            language,
            model,
            session,
            tokenizer,
            max_diff_threshold=1e-2,  # FP16なのでより緩い閾値を設定
            mean_diff_threshold=1e-3,
        )
        color = "green" if is_valid else "red"
        print(f"[bold {color}]{message}[/bold {color}]")

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
