import regex as re
from collections import Counter
from pathlib import Path

# GPT-2的预分词正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    #生成一个包含所有一字节token的初始词表
    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    #把special_tokens接在初始词表后面
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    return vocab

def split_on_special_tokens(text:str, special_tokens:list[str])-> list[str]:
    if not special_tokens:
        return [text]
    #将special_tokens转义，防止影响正则表达式
    escaped = [re.escape(token) for token in special_tokens]
    #将所有special_tokens用 | 连接起来，|表示或
    pattern = "|".join(escaped)
    #按 special tokens切分文本，不保留分隔符，并移除空串
    parts = re.split(pattern,text)

    return[p for p in parts if p != ""]

# 预分词函数
def pretokenize_text(text:str) -> list[str]:
    return re.findall(PAT, text)

# 把token变成uft-8编码的元组
def pretoken_to_byte_tuple(token:str) ->tuple[bytes, ...]:
    #把token变成utf-8编码
    token_bytes = token.encode("utf-8")

    return tuple(bytes([b]) for b in token_bytes)

# 建立频次表，建立预分词之后的单词的对应字母元组的频次表
def build_word_freqs(input_path:str, special_tokens:list[str]):
    # 创建空计数器
    word_freqs = Counter()

    # 打开文件并且载入内容
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 根据special_tokens分块
    chunks = split_on_special_tokens(text, special_tokens)

    # 分块之后，把小块内容进行预分词
    for chunk in chunks:
        pretokens = pretokenize_text(chunk)

        # 预分词之后，把单词中的每个字母都进行utf-8编码元组
        for token in pretokens:
            byte_tuple = pretoken_to_byte_tuple(token)
            # 计算byte_tuple出现了多少次
            word_freqs[byte_tuple] += 1

    return word_freqs

# 建立pair的频次表
def get_pair_counts(words_freqs):
    pair_freqs = Counter()

    # 计算每个word中字母的pair出现的次数，不会跨word计算
    for word, freq in words_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
    
    return pair_freqs