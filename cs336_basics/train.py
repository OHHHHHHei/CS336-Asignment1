import torch
import time
from cs336_basics.data import get_batch
from cs336_basics.nn import cross_entropy_loss
from cs336_basics.optimizer import gradient_clipping, lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint
from cs336_basics.logging_utils import log_metrics, save_config


def evaluate(model, eval_data, batch_size, context_length, num_eval_batches, device):
    # 将模型切换到评估模式，这样在评估过程中，模型的行为会有所不同
    # 例如某些层（如 dropout 和 batch normalization）会表现出不同的行为。
    model.eval()
    
    losses = []
    # 不需要计算梯度，因为是评估阶段，我们只需要前向传播来计算损失。
    with torch.no_grad():
        for _ in range(num_eval_batches):
            x, y = get_batch(
                dataset=eval_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())

    model.train()  # 切换回训练模式
    
    # 返回评估损失的平均值，如果 losses 列表不为空，则计算平均值，否则返回 0.0，以避免除以零的情况。
    return sum(losses) / len(losses) if losses else 0.0

def train(
    model, 
    train_data, 
    optimizer, 
    batch_size, 
    context_length, 
    num_iters,
    eval_interval,
    eval_data,
    num_eval_batches,
    max_l2_norm, 
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
    device,
    checkpoint_path=None,
    run_dir=None
    ):
    model.train()
    start_time = time.time()
    losses = []
    eval_losses = []
    learning_rates = []
    # 训练 num_iters 次，每次从训练数据中获取一个批次，计算损失，反向传播梯度，并更新模型参数。
    for iteration in range(num_iters):
      
        # 学习率调度
        lr = lr_cosine_schedule(
            iteration=iteration,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        # 更新优化器的学习率，以便在训练过程中动态调整学习率
        # 通常在预热阶段逐渐增加，在余弦衰减阶段逐渐减少。
        for group in optimizer.param_groups:
            group['lr'] = lr
        # 获得数据
        x, y = get_batch(
            dataset = train_data,
            batch_size = batch_size,
            context_length = context_length,
            device = device
        )
        # 初始化优化器的梯度为零
        optimizer.zero_grad()
        # 获取模型的输出
        logits = model(x)
        # 计算交叉熵损失
        loss = cross_entropy_loss(logits, y)
        # 进行反向传播，计算每个参数的梯度
        loss.backward()
        # 对所有参数的梯度进行裁剪，确保它们的 L2 范数不超过 max_l2_norm，以防止梯度爆炸。
        gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        # 更新模型参数
        optimizer.step()
        # 将当前的损失值添加到 losses 列表中，以便后续分析和可视化。
        losses.append(loss.item())
        # 将当前的学习率添加到 learning_rates 列表中，以便后续分析和可视化。
        learning_rates.append(lr)

        # 记录当前的训练指标，包括迭代次数、训练损失、学习率和时间等信息，以便后续分析和可视化。
        metrics = {
            "iteration": iteration + 1,
            "train_loss": loss.item(),
            "learning_rate": lr,
            "time_elapsed": time.time() - start_time
        }

        # 如果当前迭代次数是 eval_interval 的倍数，就在评估数据上评估模型的性能，并记录评估损失。
        if (iteration + 1) % eval_interval == 0:
            eval_loss = evaluate(model, eval_data, batch_size, context_length, num_eval_batches, device)
            eval_losses.append((iteration + 1, eval_loss))
            print(f"Iteration {iteration + 1}: Eval Loss = {eval_loss:.4f}")
            metrics["eval_loss"] = eval_loss
            if checkpoint_path is not None:
                save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
        # 写入日志
        if run_dir is not None:
            log_metrics(metrics, run_dir)

    return losses, learning_rates, eval_losses

