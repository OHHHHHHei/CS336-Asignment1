import regex as re
from collections.abc import Iterable

class BPEtokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]) -> None:
        # 初始化 BPEtokenizer 类，接受词汇表、合并规则和特殊标记列表作为参数。
        self.vocab = vocab
        # 创建一个字节到 ID 的映射（id_to_byte）和一个 ID 到字节的映射（byte_to_id），以便在编码和解码过程中使用。
        self.id_to_byte = vocab # 从 id 找到具体的 byte
        self.byte_to_id = {v: k for k, v in vocab.items()} # 从 byte 找到具体的 id

        self.merges = merges

        self.special_tokens = special_tokens or []

        if self.special_tokens:
            # 把special tokens 按照长度从大到小排序
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 转义特殊标记以便在正则表达式中使用，并将它们连接成一个模式字符串，以便后续使用正则表达式进行匹配。
            special_tokens_pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)
            # 把特殊标记的模式编译成一个正则表达式对象（special_tokens_regex），以便在后续的文本处理过程中使用它来识别和处理特殊标记。
            self.special_tokens_regex = re.compile(special_tokens_pattern)
        else:
            self.special_tokens_regex = None
        # 这是GPT-2的官方分词器使用的正则表达式模式，用于匹配文本中的不同类型的标记，包括特殊标记、单词、数字、非空白字符和空白字符。这个模式确保了在分词过程中能够正确识别和处理各种类型的文本元素。
        self.gpt2_byte_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:

        if not text:
            # 如果输入文本为空，则返回一个空列表。
            return []

        if not self.special_tokens:
            # 如果没有特殊标记，则直接使用 GPT-2 的字节模式正则表达式来分割输入文本，并将每个标记转换为对应的 ID，最后返回一个包含这些 ID 的列表。
            return self._encode_text_segment(text)

        tokens = []

        # 标记目前已经处理到输入文本的哪个位置了，初始值为0。
        last_pos = 0

        # 找到text中所有的 special tokens
        for match in self.special_tokens_regex.finditer(text):
            # 取text中上次处理的位置到 special token 开始位置之间的文本
            pre_text = text[last_pos:match.start()]

            if pre_text:
                # 直接把这段文本按照GPT-2的分词方式进行编码，并将得到的 ID 添加到 tokens 列表中。
                tokens.extend(self._encode_text_segment(pre_text))
            # 找出 special token
            special_token = match.group()
            # 把 special token 转换为对应的 ID，并将其添加到 tokens 列表中。
            tokens.append(self.byte_to_id[special_token.encode("utf-8")])
            # 更新 last_pos 以指向当前 special token 的结束位置，为下一次迭代做准备。
            last_pos = match.end()

        # 处理最后一个 special token 之后的文本
        remaining_text = text[last_pos:]
        if remaining_text:
            # 直接编码放进tokens列表中
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens
    
    # 具体对 text 进行 encode 的函数，输入是文本，输出是对应的 ID 列表
    def _encode_text_segment(self, text: str) -> list[int]:
        
        ids = []
        # 把接受到的 text 进行预分词
        pre_tokens = self.gpt2_byte_pattern.findall(text)
        # 遍历每个 pre_token
        for token in pre_tokens:
            # 把每个单词用 UTF-8 编码成字节，然后把每个单词中的字母转换成对应的独立字节，并将这些字节存储在 byte_parts 列表中，以便后续的 BPE 合并操作。
            byte_parts = [bytes([b]) for b in token.encode("utf-8")]
            
            while len(byte_parts) > 1:

                best_pair = None
                min_rank = float('inf')

                for i in range(len(byte_parts) - 1):
                    # 扫描所有相邻的 pair
                    pair = (byte_parts[i], byte_parts[i + 1])
                    # 如果这个 pair 在 merge 当中
                    if pair in self.merges:
                        # 找到这个 pair 对应的优先级
                        rank = self.merges.index(pair)
                        # 如果这个 pair 的优先级比当前找到的 best_pair 更高（即 rank 更小），就更新 best_pair 和 min_rank。
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break
                
                new_byte_parts = []
                i = 0

                while i < len(byte_parts):
                    # 如果找到了 best_pair，就把它们合并成一个新的字节
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i + 1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2 # 跳过下一个字节，因为它已经被合并了
                    else:
                        # 如果没有 best——pair，就把当前的字节直接添加到 new_byte_parts 列表中
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                # 更新 byte_parts 准备进行下一轮合并
                byte_parts = new_byte_parts

            # 合并完成之后
            for byte_part in byte_parts:
                # 把每个最终的 byte_part 转换为对应的 ID，并将这些 ID 添加到 ids 列表中，最后返回这个列表。
                ids.append(self.byte_to_id[byte_part])

        return ids