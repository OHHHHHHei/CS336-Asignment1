# TinyStories 文本生成

## 实验设置

- Checkpoint：`tinystories_base/checkpoint.pt`，迭代次数 20,000
- Prompt：`Hello, she said`
- Temperature：`0.8`
- Top-p：`0.9`
- 随机种子：`42`
- 生成长度：包含 prompt 在内共 113 个 token
- 停止条件：首次生成 `<|endoftext|>` token

## 生成文本

Hello, she said, "Hi, I am Lily. Do you want to play with me?" The snake said, "Yes, I want to play too!"

Lily and the snake played all day. They ran, jumped, and laughed. But then, something unexpected happened. The snake changed into a big, friendly dog! The dog said, "I am a magic dog. I can make your wishes come true." Lily was so surprised and happy. She wished for a big ice cream cone. The magic dog made her wish come true.

`<|endoftext|>`

## 结果分析

对于 TinyStories 这一数据领域，样例的语言流畅度尚可：它构成了一个完整的短故事，在多个句子之间保持了主要人物和动物的一致，并在 EOS token 处正常结束。文本仍然比较简单且有一定重复，这是一个约 17M 参数的小模型在单一儿童故事语料上训练后的合理表现。

影响生成结果的主要因素有：

1. Temperature 和 top-p 控制 token sampling 的随机性。提高 temperature 或放宽 top-p 通常会增加多样性，但也可能降低故事连贯性；较低的取值更稳妥，但更容易重复。
2. Checkpoint 质量和训练数据决定模型学到的词汇、语法和故事模式。更晚或调参更好的 checkpoint 可能带来更流畅的文本，而 TinyStories 数据领域本身会限制内容和写作风格。
3. Prompt 和上下文长度也会影响续写。更具体的 prompt 可以引导主题，但 256-token 的 context window 限制了模型能够利用的历史文本长度。
