# CS336 Assignment 1

这是一个从零实现 Transformer 语言模型核心组件的课程作业项目，包含：

- 字节级 BPE 分词器训练与加载
- 语料流式预处理（文本转 token 二进制）
- Transformer LM 训练（含学习率调度、梯度裁剪、断点续训）
- 交互式推理脚本
- 若干消融实验开关（RMSNorm / RoPE / FFN 类型 / Norm 位置）

## 项目结构

```text
cs336_basics/
  checkpointing.py
  data.py
  inference.py
  losses.py
  main_train.py
  nn.py
  optimizer.py
  preprocess.py
  scheduler.py
  sgd.py
  tokenizer.py
  train_bpe.py
tests/
  adapters.py
bpe.ipynb
```

## 环境准备

建议使用 Python 3.10+。

1. 创建并激活虚拟环境（可选）
2. 安装依赖：

```bash
pip install torch numpy regex wandb jaxtyping
```

如果你要运行测试，建议额外安装：

```bash
pip install pytest
```

## 快速开始

### 1) 训练 BPE 分词器

当前脚本默认读取：

- 输入语料：`data/TinyStoriesV2-GPT4-train.txt`
- 输出目录：`data/TinyStoriesV2-GPT4-train`
- 目标词表：10000

运行：

```bash
python -m cs336_basics.train_bpe
```

输出文件：

- `vocab.json`
- `merges.txt`

### 2) 将文本语料预处理为二进制 token

当前 `preprocess.py` 默认配置为：

- 分词器目录：`data/TinyStoriesV2-GPT4-train`
- 输入文本：`data/TinyStoriesV2-GPT4-valid.txt`
- 输出二进制：`data/TinyStoriesV2-GPT4-valid.bin`

运行：

```bash
python -m cs336_basics.preprocess
```

说明：该脚本使用流式处理与分批写入，适合较大语料。

### 3) 训练 Transformer 语言模型

`main_train.py` 需要显式传入训练/验证二进制文件路径。

示例：

```bash
python -m cs336_basics.main_train \
  --train_data_path data/TinyStoriesV2-GPT4-train.bin \
  --valid_data_path data/TinyStoriesV2-GPT4-valid.bin \
  --vocab_size 10000 \
  --batch_size 32 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 2048 \
  --max_iters 10000 \
  --out_dir out
```

可选消融参数：

- `--no_rms_norm`：关闭 RMSNorm
- `--norm_mode pre|post`：选择 pre-norm / post-norm
- `--no_rope`：关闭 RoPE
- `--ffn_type swiglu|silu`：切换 FFN 类型

训练期间会：

- 定期打印 train/val loss
- 记录到 Weights & Biases（项目名 `cs336-assignment1`）
- 在 `out/ckpt.pt` 自动保存检查点

### 4) 交互式推理

```bash
python -m cs336_basics.inference \
  --checkpoint_path out/ckpt.pt \
  --tokenizer_dir data/TinyStoriesV2-GPT4-train \
  --vocab_size 10000 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_new_tokens 100
```

注意：推理时模型结构参数必须与训练时一致，否则会出现权重加载失败。

## 测试

项目包含 `tests/adapters.py` 作为适配入口。

如果你有完整测试集，可使用：

```bash
pytest -q
```

## 常见问题

1. 权重加载报错（shape mismatch）
   - 检查 `inference.py` 的模型超参数是否与训练完全一致。

2. 找不到分词器文件
   - 确认 `tokenizer_dir` 下存在 `vocab.json` 与 `merges.txt`。

3. 显存不足
   - 减小 `batch_size`、`context_length` 或 `d_model`。

4. WandB 登录问题
   - 先执行 `wandb login`，或在离线环境中关闭 wandb 相关逻辑。

## 说明

- 该仓库用于课程学习与实验复现。
- 如需提交作业，请按课程要求补充实验结果、曲线图和分析结论。
