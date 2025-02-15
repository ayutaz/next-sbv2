import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import get_config
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT

config = get_config()
# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()
# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

# ------------------------------------------------------------------
# グローバル変数として、ロード済みの tokenizer, model をキャッシュ
# ------------------------------------------------------------------
GLOBAL_TOKENIZER = None
GLOBAL_MODEL = None
MODEL_NAME = "sbintuitions/modernbert-ja-130m"

def setup_bert(device: str = "cpu"):
    """
    tokenizer, model を一度だけロードし、グローバル変数にキャッシュする。
    すでにロード済みなら何もしない。
    """
    global GLOBAL_TOKENIZER, GLOBAL_MODEL
    if GLOBAL_TOKENIZER is None or GLOBAL_MODEL is None:
        print(f"[setup_bert] Loading tokenizer+model: {MODEL_NAME} ...")
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        GLOBAL_MODEL = AutoModel.from_pretrained(MODEL_NAME).to(device)
        GLOBAL_MODEL.eval()

    return GLOBAL_TOKENIZER, GLOBAL_MODEL

class ModernBertCharTokenizer:
    """
    ModernBERT (sbintuitions/modernbert-ja-130m) を使いつつ、
    '1文字=1トークン' として扱うためのラッパークラス。
    """
    def __init__(self, base_tokenizer):
        # base_tokenizer は AutoTokenizer.from_pretrained(...) の結果
        self.base_tokenizer = base_tokenizer

    def encode_plus_as_chars(
        self,
        text: str,
        device: str = "cpu",
        add_special_tokens: bool = True
    ):
        """
        テキストを 1文字ずつに分割 → base_tokenizer.encode_plus(..., is_split_into_words=True)
        で BERT に入れられる形式を返す
        """
        # 例: punctuation除去や正規化は呼び出し元で済ませるなら省略可
        # ここでは最低限 normalize_text 程度にとどめる
        text = normalize_text(text)

        # 1文字ごとに分割
        char_list = list(text)  # 例: "こんにちは"→["こ","ん","に","ち","は"]

        # base_tokenizer でエンコード
        encoding = self.base_tokenizer.encode_plus(
            char_list,
            is_split_into_words=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        # shapeは (1, seq_len+特別トークン) のはず
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        return input_ids, attention_mask

def extract_bert_feature_char_level(
    text: str,
    word2ph: list[int],
    device: str = "cpu"
) -> torch.Tensor:
    """
    1文字=1トークンで BERT に通した結果を「(hidden_size, sum(word2ph))」に展開して返す。
    """
    tokenizer, model = setup_bert(device)

    # ラッパークラスを生成
    char_tokenizer = ModernBertCharTokenizer(tokenizer)

    # BERTに通す
    input_ids, attention_mask = char_tokenizer.encode_plus_as_chars(
        text, device=device, add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # hidden_states の末尾2層を結合
        cat_h = torch.cat(outputs.hidden_states[-2:], dim=-1)  # (1, seq_len, hidden*2)
        cat_h = cat_h[0]  # => (seq_len, hidden_size_of_concat)

    # [CLS], [SEP] 分を除去
    # seq_len_total = cat_h.shape[0]
    # => if add_special_tokens=True, 先頭1個+末尾1個 => 2個除去
    cat_h_no_sp = cat_h[1:-1]  # => (seq_len-2, hidden_size_of_concat)
    char_len = cat_h_no_sp.shape[0]

    # word2ph の長さ == char_len である必要がある
    if len(word2ph) != char_len:
        print(
            f"[extract_bert_feature_char_level] mismatch: "
            f"char_len={char_len}, len(word2ph)={len(word2ph)} => skip"
        )
        return None

    # 転置 => (hidden_size_of_concat, char_len)
    cat_h_t = cat_h_no_sp.transpose(0, 1).contiguous()

    # 繰り返し展開 => (hidden_size, sum(word2ph))
    phone_level_list = []
    for i, rep in enumerate(word2ph):
        v_i = cat_h_t[:, i]  # (hidden_size_of_concat,)
        repeated = v_i.unsqueeze(1).repeat(1, rep)
        phone_level_list.append(repeated)

    phone_level_emb = torch.cat(phone_level_list, dim=1)  # => (hidden_size, sum(word2ph))

    return phone_level_emb

def process_line(x: tuple[str, bool]):
    """
    train.list の1行 (wav_path|spk|lang|text|phones|tones|word2ph) を受け取り、
    "1文字=1トークン" BERT 特徴量 (.bert.pt) を生成する例。
    """
    line, add_blank = x
    print("[Step4-debug] original line =>", line.strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 行をパース
    try:
        wav_path, spk, language_str, raw_text, phones_str, tone_str, word2ph_str = \
            line.strip().split("|")
    except ValueError:
        print("[Step4-debug] parse error => skip line")
        return

    # 既に .bert.pt があればスキップ
    bert_path = wav_path.replace(".wav", ".bert.pt").replace(".WAV", ".bert.pt")
    try:
        _ = torch.load(bert_path)
        print(f"[Step4-debug] already have valid .bert.pt => skip: {bert_path}")
        return
    except Exception:
        pass

    # word2ph は phone の各文字に音素数が割り当てられるリスト
    # ただし "add_blank=True" で intersperse するなら2倍になる等、
    # phone_list の先頭末尾に "_" を挿入するならさらに +2 など、step4 と学習時を合わせる必要がある
    # ここでは単純に "phones_str.split()" と "word2ph_str.split()" を対応させる想定
    phone_list = phones_str.split(" ")
    word2ph = [int(x) for x in word2ph_str.split(" ")]

    # BERT埋め込みを抽出 (1文字=1トークン方式)
    bert_mat = extract_bert_feature_char_level(
        text=raw_text,
        word2ph=word2ph,
        device=device
    )
    if bert_mat is None:
        print("[Step4-debug] mismatch => skip")
        return

    # shapeチェック
    hidden_size, phone_len = bert_mat.shape
    print(f"[Step4-debug] got bert shape => {bert_mat.shape}")

    # 実際に .bert.pt として保存
    torch.save(bert_mat, bert_path)
    print(f"[Step4-debug] saved => {bert_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        # pyopenjtalkの別ワーカー化により、並列処理でエラーが出る場合があるので
        num_processes = 1
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(process_line, zip(lines, add_blank)),
                    total=len(lines),
                    file=SAFE_STDOUT,
                )
            )
    print("[Step4-debug] bert.pt generation is completed! total:", len(lines))
