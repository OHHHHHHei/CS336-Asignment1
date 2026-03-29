"""CS336 Assignment 1 的 byte-level BPE 训练实现。

这份实现刻意保持结构清晰，方便按训练流程理解：

1. 初始化词表：包含 256 个基础字节和所有 special tokens。
2. 用 GPT-2 的正则做 pre-tokenization。
3. 把每个 pre-token 表示为由单字节 ``bytes`` 组成的 tuple。
4. 统计相邻 byte pair 的频次。
5. 反复合并当前最常见的 pair，直到达到目标词表大小。

为了通过速度测试，这里最终采用了 pair 频次缓存：
每次 merge 后，不再全量重算所有 pair，而是只增量更新受影响部分。
"""

import regex as re
from collections import Counter


# 作业 handout / tiktoken 中给出的 GPT-2 预分词正则。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 为了提高可读性，给核心数据结构起一些类型别名。
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordFreqs = Counter[Word]
PairFreqs = Counter[Pair]
WordPairCounts = dict[Word, Counter[Pair]]


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """初始化 byte-level BPE 训练使用的词表。

    词表一定从 256 个基础字节开始，然后把用户提供的
    special tokens 依次追加到后面。
    """

    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    return vocab


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """在预分词之前，先按 special tokens 切分原始文本。

    像 ``<|endoftext|>`` 这样的 special token 表示边界，merge
    不能跨越这些边界。因此训练时我们要先把它们从普通文本流中切开，
    只保留它们之间的普通文本片段。
    """

    if not special_tokens:
        return [text]

    escaped = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped)
    parts = re.split(pattern, text)

    # 去掉前后 special token 造成的空字符串片段。
    return [part for part in parts if part != ""]


def pretokenize_text(text: str) -> list[str]:
    """对一个普通文本片段应用 GPT-2 风格的预分词。"""

    return re.findall(PAT, text)


def pretoken_to_byte_tuple(token: str) -> Word:
    """把一个 pre-token 字符串转成由单字节 token 组成的 tuple。

    例如：
        " low" -> (b" ", b"l", b"o", b"w")
    """

    token_bytes = token.encode("utf-8")
    return tuple(bytes([b]) for b in token_bytes)


def build_word_freqs(input_path: str, special_tokens: list[str]) -> WordFreqs:
    """构建 BPE 训练使用的 pre-token 频次表。

    key 是由字节 token 构成的 tuple，value 是它在语料中的出现次数。
    这正是 handout 推荐的紧凑表示方式。
    """

    word_freqs: WordFreqs = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_on_special_tokens(text, special_tokens)

    for chunk in chunks:
        for token in pretokenize_text(chunk):
            word_freqs[pretoken_to_byte_tuple(token)] += 1

    return word_freqs


def get_pair_counts(word_freqs: WordFreqs) -> PairFreqs:
    """朴素地统计整个当前语料状态中的相邻 pair 频次。

    这个函数主要用于理解算法流程。真正优化后的训练主循环
    不会在每一轮都重新调用它，而是使用缓存做增量更新。
    """

    pair_freqs: PairFreqs = Counter()

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


def get_best_pair(pair_counts: PairFreqs) -> Pair | None:
    """返回当前下一轮最应该 merge 的 pair。

    规则是：
    1. 先选频次最大的 pair。
    2. 如果频次并列，则按作业要求选择字典序更大的 pair。
    """

    if not pair_counts:
        return None

    best_pair: Pair | None = None
    max_count = -1

    for pair, freq in pair_counts.items():
        if freq > max_count:
            max_count = freq
            best_pair = pair
        elif freq == max_count and pair > best_pair:
            best_pair = pair

    return best_pair


def merge_word(word: Word, best_pair: Pair) -> Word:
    """对单个 word 应用一次 BPE merge。

    实现方式是从左到右扫描，并进行不重叠 merge。例如：
        (b"a", b"a", b"a") 和 pair (b"a", b"a")
    会得到：
        (b"aa", b"a")
    """

    merged: list[bytes] = []
    i = 0
    n = len(word)

    while i < n:
        if i < n - 1 and (word[i], word[i + 1]) == best_pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)


def apply_merge(word_freqs: WordFreqs, best_pair: Pair) -> WordFreqs:
    """朴素地把一次 merge 应用到所有 word 上。

    和 ``get_pair_counts`` 一样，这个函数主要保留下来帮助理解；
    真正用于通过速度测试的是下面的缓存增量更新版本。
    """

    new_word_freqs: WordFreqs = Counter()

    for word, freq in word_freqs.items():
        new_word = merge_word(word, best_pair)
        new_word_freqs[new_word] += freq

    return new_word_freqs


def count_pairs_in_word(word: Word) -> Counter[Pair]:
    """统计单个 word 内部所有相邻 pair 的局部频次。"""

    local_pair_counts: Counter[Pair] = Counter()
    for i in range(len(word) - 1):
        local_pair_counts[(word[i], word[i + 1])] += 1
    return local_pair_counts


def build_pair_data(word_freqs: WordFreqs) -> tuple[WordPairCounts, PairFreqs]:
    """为优化后的训练主循环初始化 pair 缓存数据。

    返回两个对象：
    1. ``word_pair_counts``：
       记录每个 word 内部各个相邻 pair 的局部频次。
    2. ``pair_freqs``：
       记录整个语料上的全局 pair 频次，已经乘上了每个 word 的语料频次。
    """

    word_pair_counts: WordPairCounts = {}
    pair_freqs: PairFreqs = Counter()

    for word, freq in word_freqs.items():
        local_pair_counts = count_pairs_in_word(word)
        word_pair_counts[word] = local_pair_counts

        for pair, count in local_pair_counts.items():
            pair_freqs[pair] += count * freq

    return word_pair_counts, pair_freqs


def apply_merge_and_count_pairs(
    word_freqs: WordFreqs,
    word_pair_counts: WordPairCounts,
    pair_freqs: PairFreqs,
    best_pair: Pair,
) -> tuple[WordFreqs, WordPairCounts, PairFreqs]:
    """执行一次 merge，并增量更新缓存的 pair 频次。

    这是相对于朴素实现最关键的优化点。

    我们不再在每一轮 merge 后全量重算所有 pair，而是只更新
    真正受 ``best_pair`` 影响的那些 word：

    - 如果某个 word 不包含 ``best_pair``，就直接原样复用。
    - 如果某个 word 包含 ``best_pair``，就：
      1. 先从全局缓存里减掉这个旧 word 的 pair 贡献；
      2. 对这个 word 执行 merge；
      3. 统计 merge 后新 word 的局部 pair；
      4. 再把新 word 的贡献加回全局缓存。
    """

    new_word_freqs: WordFreqs = Counter()
    new_word_pair_counts: WordPairCounts = {}
    new_pair_freqs: PairFreqs = pair_freqs.copy()

    for word, freq in word_freqs.items():
        local_pair_counts = word_pair_counts[word]

        # 不受当前 best_pair 影响的 word 可以直接复用。
        if best_pair not in local_pair_counts:
            new_word_freqs[word] += freq
            new_word_pair_counts[word] = local_pair_counts
            continue

        # 先把旧 word 对全局 pair 频次的贡献减掉。
        for pair, count in local_pair_counts.items():
            new_pair_freqs[pair] -= count * freq
            if new_pair_freqs[pair] == 0:
                del new_pair_freqs[pair]

        # 对旧 word 做 merge，并统计新 word 的局部 pair 频次。
        new_word = merge_word(word, best_pair)
        new_word_freqs[new_word] += freq

        # 多个旧 word 可能 merge 成同一个新 word，所以如果已经算过
        # 这个新 word 的局部 pair 频次，就直接复用。
        new_local_pair_counts = new_word_pair_counts.get(new_word)
        if new_local_pair_counts is None:
            new_local_pair_counts = count_pairs_in_word(new_word)
            new_word_pair_counts[new_word] = new_local_pair_counts

        # 再把新 word 对全局 pair 频次的贡献加回去。
        for pair, count in new_local_pair_counts.items():
            new_pair_freqs[pair] += count * freq

    return new_word_freqs, new_word_pair_counts, new_pair_freqs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """训练 byte-level BPE tokenizer，并返回 ``(vocab, merges)``。

    - ``vocab``：最终的 token_id -> bytes 映射
    - ``merges``：按创建顺序记录的 merge 列表
    """

    vocab = init_vocab(special_tokens)
    word_freqs = build_word_freqs(input_path, special_tokens)

    merges: list[Pair] = []
    next_id = len(vocab)

    # 初始化优化版训练循环所需的 pair 缓存。
    word_pair_counts, pair_freqs = build_pair_data(word_freqs)

    while len(vocab) < vocab_size:
        best_pair = get_best_pair(pair_freqs)
        if best_pair is None:
            break

        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        word_freqs, word_pair_counts, pair_freqs = apply_merge_and_count_pairs(
            word_freqs, word_pair_counts, pair_freqs, best_pair
        )

    return vocab, merges
