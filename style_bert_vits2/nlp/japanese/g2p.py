from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

# --- 文字トークンだけ返す簡易版 ---
def g2p(norm_text: str, *_, **__) -> tuple[list[str], list[int], list[int]]:
    """
    OpenJTalk を使わず、BERT トークン列を直接 TTS に渡す。
    - phones: ['_'] + tokens + ['_']
    - tones : 全て 0 (ダミー)
    - word2ph: len(phones) と同じ長さで全部 1
    """
    tokens = bert_models.load_tokenizer(Languages.JP).tokenize(norm_text)
    phones  = tokens
    tones   = [0] * len(phones)
    word2ph = [1] * len(phones)
    return phones, tones, word2ph