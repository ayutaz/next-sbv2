from __future__ import annotations

from typing import Any, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.utils import get_onnx_device_options

if TYPE_CHECKING:
    import torch


# ------------------------------------------------------------------
#  PyTorch 推論
# ------------------------------------------------------------------
def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> "torch.Tensor":
    """
    BERT 特徴量を PyTorch で抽出（char-only 版）。

    * phones と同数の token を返すので g2p で付与した PAD は不要。
    """
    import torch

    tokenizer = bert_models.load_tokenizer(Languages.JP)
    tokens = tokenizer.tokenize(text)           # g2p と同一のトークン列
    inputs = tokenizer(text, return_tensors="pt")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = bert_models.load_model(Languages.JP, device_map=device)
    bert_models.transfer_model(Languages.JP, device)
    for k in inputs:
        inputs[k] = inputs[k].to(device)        # type: ignore

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = torch.cat(outputs.hidden_states[-3:-2], -1)[0].cpu()  # CLS + tokens + SEP
        hs = hs[1 : 1 + len(tokens)]                               # token 部分だけ抜き出す

        # assist_text ありの場合
        style_mean = None
        if assist_text:
            st_inputs = tokenizer(assist_text, return_tensors="pt")
            for k in st_inputs:
                st_inputs[k] = st_inputs[k].to(device)  # type: ignore
            st_out = model(**st_inputs, output_hidden_states=True)
            st_hs = torch.cat(st_out.hidden_states[-3:-2], -1)[0].cpu()
            style_mean = st_hs.mean(0)

    assert len(word2ph) == len(tokens), (len(word2ph), len(tokens))

    feat_chunks = []
    for i, n_rep in enumerate(word2ph):
        base = hs[i].repeat(n_rep, 1)
        if assist_text and style_mean is not None:
            base = base * (1 - assist_text_weight) + style_mean.repeat(n_rep, 1) * assist_text_weight
        feat_chunks.append(base)

    phone_level_feature = torch.cat(feat_chunks, 0)   # [phones, hidden]
    return phone_level_feature.T                      # [hidden, phones]


# ------------------------------------------------------------------
#  ONNX 推論
# ------------------------------------------------------------------
def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    BERT 特徴量を ONNXRuntime で抽出（char-only 版）。
    """
    tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
    tokens = tokenizer.tokenize(text)

    session = onnx_bert_models.load_model(Languages.JP, onnx_providers)
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    dev_type, dev_id, run_opt = get_onnx_device_options(session, onnx_providers)

    def _run_onnx(txt: str) -> NDArray[Any]:
        ins = tokenizer(txt, return_tensors="np")
        tensors = [
            ins["input_ids"].astype(np.int64),      # type: ignore
            ins["attention_mask"].astype(np.int64), # type: ignore
        ]
        io_bind = session.io_binding()
        for name, val in zip(input_names, tensors):
            ov = onnxruntime.OrtValue.ortvalue_from_numpy(val, dev_type, dev_id)
            io_bind.bind_ortvalue_input(name, ov)
        io_bind.bind_output(output_name, dev_type)
        session.run_with_iobinding(io_bind, run_options=run_opt)
        return io_bind.get_outputs()[0].numpy()

    res = _run_onnx(text)[1 : 1 + len(tokens)]        # drop CLS

    style_mean = None
    if assist_text:
        style = _run_onnx(assist_text)[1:-1]          # drop CLS/SEP
        style_mean = style.mean(0)

    assert len(word2ph) == len(tokens), (len(word2ph), len(tokens))

    chunks: list[NDArray[Any]] = []
    for i, n_rep in enumerate(word2ph):
        base = np.tile(res[i], (n_rep, 1))
        if assist_text and style_mean is not None:
            base = base * (1 - assist_text_weight) + np.tile(style_mean, (n_rep, 1)) * assist_text_weight
        chunks.append(base)

    phone_level_feature = np.concatenate(chunks, axis=0)   # [phones, hidden]
    return phone_level_feature.T                           # [hidden, phones]
