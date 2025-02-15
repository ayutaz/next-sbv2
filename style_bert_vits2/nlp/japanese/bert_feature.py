from __future__ import annotations

from typing import Any, Sequence, Union
from typing import Optional

import numpy as np
import onnxruntime
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp import onnx_bert_models
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata, g2p, adjust_word2ph
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.utils import get_onnx_device_options


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    日本語のテキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str):
            元テキスト（train.list に書かれているようなもの）。
            ただし実際には記号除去などをしてから BERT にかける。
        word2ph (list[int]):
            元のテキストの各文字に音素が何個割り当てられるかを表すリスト。
            先頭と末尾に「1」が加えられている形式の場合が多い。
        device (str):
            推論に利用するデバイス ("cuda" / "cpu" 等)。
        assist_text (Optional[str], optional):
            補助テキスト。ある場合のみ抽出結果にブレンドする。
        assist_text_weight (float, optional):
            補助テキストの重み。 Defaults to 0.7.

    Returns:
        torch.Tensor:
            shape = (hidden_size, sum(word2ph)) 程度のテンソル
    """

    # ----------------------------------------------------------------
    # 1. text -> pyopenjtalk で読めない文字を除去したもの (punctuation や空白等) にする
    #    → text_to_sep_kata(..., raise_yomi_error=False) で取り除き、
    #      結合して "pure_text" を作る
    # ----------------------------------------------------------------
    pure_text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    # assist_text についても同じことをする
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    # ----------------------------------------------------------------
    # 2. word2ph の長さが合わない場合に備え、チェックをする
    #    → g2p() と adjust_word2ph() を使って整合性を取り直す
    # ----------------------------------------------------------------
    required_len = len(pure_text) + 2
    if len(word2ph) != required_len:
        print(
            f"[extract_bert_feature] Mismatch: len(word2ph)={len(word2ph)}, "
            f"but expected {required_len}. Attempt to fix..."
        )
        # 2-1. g2p() する:
        normed = normalize_text(pure_text)
        phones, tones, g2p_word2ph, _ = g2p(normed, use_jp_extra=True, raise_yomi_error=False)

        # 2-2. adjust_word2ph() で既存 word2ph に近い形を再計算、または仕方なく phones=given_phone として同一視
        try:
            new_word2ph = adjust_word2ph(
                g2p_word2ph,  # generated_word2ph
                phones,       # generated_phone
                phones,       # given_phone（なければ同じでOK）
            )
            word2ph = new_word2ph
            print(
                f"[extract_bert_feature] Adjusted word2ph => length={len(word2ph)}, sum={sum(word2ph)}"
            )
        except AssertionError as e:
            print("[extract_bert_feature] adjust_word2ph failed:", e)
            # 最終手段: return 0テンソル or スキップ
            return torch.zeros(1, dtype=torch.float32)

    # device が "cuda" でも、実際に GPU が無い場合は "cpu" に変える
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ----------------------------------------------------------------
    # 3. BERT モデルをロード、テキストをトークナイズしてベクトルを取得
    # ----------------------------------------------------------------
    model = bert_models.load_model(Languages.JP, device_map=device)
    bert_models.transfer_model(Languages.JP, device)

    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.JP)
        inputs = tokenizer(pure_text, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        outputs = model(**inputs, output_hidden_states=True)
        # [-3:-2] → 隠れ層の最後から3番目だけ取り出す例（用途に応じて要調整）
        cat_h = torch.cat(outputs["hidden_states"][-3:-2], dim=-1)
        # shape = (batch=1, seq_len, hidden_size_concat)
        res_cat = cat_h[0].cpu()  # shape=(seq_len, hidden_size)

        style_res_mean = None
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for k, v in style_inputs.items():
                style_inputs[k] = v.to(device)
            style_out = model(**style_inputs, output_hidden_states=True)
            style_cat_h = torch.cat(style_out["hidden_states"][-3:-2], dim=-1)[0].cpu()
            style_res_mean = style_cat_h.mean(dim=0)  # shape=(hidden_size,)

    seq_len = res_cat.shape[0]
    total_phone = sum(word2ph)  # phone の合計音素数

    # ----------------------------------------------------------------
    # 4. BERT の seq_len と、word2ph の要素数が食い違う場合の対処
    #    → “音素数” が多いのに BERTトークン数が少ない場合など
    # ----------------------------------------------------------------
    if seq_len < len(word2ph):
        print(f"[extract_bert_feature] Warning: seq_len={seq_len} < len(word2ph)={len(word2ph)}. "
              "Truncating word2ph to match seq_len.")
        # 末尾を削る (先頭末尾が 1 の場合などインデックス注意)
        word2ph = word2ph[:seq_len]  # あまり厳密に調整しない簡易実装
        # sum(word2ph) も連動して減る

    # さらに “sum(word2ph)” と “seq_len” が大きく食い違うなら → BERT埋め込みをコピーorカットする 例
    # ここでは末尾コピーで合わせる
    if seq_len < sum(word2ph):
        needed = sum(word2ph) - seq_len
        # 末尾をコピーして埋める
        last_token = res_cat[-1:].clone()  # shape=(1, hidden_size)
        extra = last_token.repeat(needed, 1)  # shape=(needed, hidden_size)
        res_cat = torch.cat([res_cat, extra], dim=0)  # shape=(sum(word2ph), hidden_size)
    elif seq_len > sum(word2ph):
        # BERT が余分に長い → 末尾カット
        res_cat = res_cat[:sum(word2ph), :]

    # ----------------------------------------------------------------
    # 5. word2ph に沿って (seq_len, hidden_size) → (sum(word2ph), hidden_size) の配列を生成
    #    assist_text_weight でブレンド
    # ----------------------------------------------------------------
    phone_level_feature = []
    offset = 0
    for wlen in word2ph:
        if offset >= res_cat.size(0):
            # 安全策: 万一 overrun したら break or 末尾コピー
            # ここでは break とする
            break
        vec = res_cat[offset]  # shape=(hidden_size,)
        offset += 1

        if style_res_mean is not None:
            merged = (
                vec.unsqueeze(0) * (1 - assist_text_weight)
                + style_res_mean.unsqueeze(0) * assist_text_weight
            )  # shape=(1, hidden_size)
            repeat_feature = merged.repeat(wlen, 1)  # shape=(wlen, hidden_size)
        else:
            repeat_feature = vec.unsqueeze(0).repeat(wlen, 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # (sum(word2ph), hidden_size)
    return phone_level_feature.T  # (hidden_size, sum(word2ph))



def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    日本語のテキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    # トークナイザーとモデルの読み込み
    tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
    session = onnx_bert_models.load_model(
        language=Languages.JP,
        onnx_providers=onnx_providers,
    )
    input_names = [input.name for input in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # 入力テンソルの転送に使用するデバイス種別, デバイス ID, 実行オプションを取得
    device_type, device_id, run_options = get_onnx_device_options(session, onnx_providers)  # fmt: skip

    # 入力をテンソルに変換
    inputs = tokenizer(text, return_tensors="np")
    input_tensor = [
        inputs["input_ids"].astype(np.int64),  # type: ignore
        inputs["attention_mask"].astype(np.int64),  # type: ignore
    ]
    # 推論デバイスに入力テンソルを割り当て
    ## GPU 推論の場合、device_type + device_id に対応する GPU デバイスに入力テンソルが割り当てられる
    io_binding = session.io_binding()
    for name, value in zip(input_names, input_tensor):
        gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
            value, device_type, device_id
        )
        io_binding.bind_ortvalue_input(name, gpu_tensor)
    # text から BERT 特徴量を抽出
    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)
    res = io_binding.get_outputs()[0].numpy()

    style_res_mean = None
    if assist_text:
        # 入力をテンソルに変換
        style_inputs = tokenizer(assist_text, return_tensors="np")
        style_input_tensor = [
            style_inputs["input_ids"].astype(np.int64),  # type: ignore
            style_inputs["attention_mask"].astype(np.int64),  # type: ignore
        ]
        # 推論デバイスに入力テンソルを割り当て
        ## GPU 推論の場合、device_type + device_id に対応する GPU デバイスに入力テンソルが割り当てられる
        io_binding = session.io_binding()  # IOBinding は作り直す必要がある
        for name, value in zip(input_names, style_input_tensor):
            gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
                value, device_type, device_id
            )
            io_binding.bind_ortvalue_input(name, gpu_tensor)
        # assist_text から BERT 特徴量を抽出
        io_binding.bind_output(output_name, device_type)
        session.run_with_iobinding(io_binding, run_options=run_options)
        style_res = io_binding.get_outputs()[0].numpy()
        style_res_mean = np.mean(style_res, axis=0)

    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(res[i], (word2phone[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2phone[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
