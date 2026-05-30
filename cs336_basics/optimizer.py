import math
import torch


def gradient_clipping(parameters, max_l2_norm):
    # 把所有参数的梯度收集到一个列表中，前提是这些参数必须有梯度（即 p.grad 不为 None）。
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    # 初始化 total_norm_squared 为一个标量张量，初始值为 0，设备与第一个梯度相同
    total_norm_squared = torch.zeros((), device=grads[0].device)

    # 累加所有参数的梯度的 L2 范数的平方
    for grad in grads:
        total_norm_squared += torch.sum(grad ** 2)

    # 计算总的 L2 范数
    total_norm = torch.sqrt(total_norm_squared)

    # 如果总的 L2 范数超过了指定的 max_l2_norm，我们就需要对所有梯度进行缩放，使得它们的 L2 范数不超过 max_l2_norm。
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)  # 添加一个小的常数以避免除以零
        for grad in grads:
            grad.mul_(scale)


def lr_cosine_schedule(
        iteration: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    # 如果在预热阶段，就线性增长到最大学习率
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    # 如果在余弦衰减阶段，就按照余弦函数衰减学习率
    elif iteration <= cosine_cycle_iters:
        cosine_progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    # 否则就保持最小学习率
    else:
        return min_learning_rate

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # 对参数组进行迭代，每个参数组可能有不同的学习率、动量等超参数。
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 对参数组中的每个参数进行迭代，更新它们的值。
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                # state 是一个字典，用于存储每个参数的状态信息
                # 包括时间步 t、动量 m 和二阶矩 v。
                state = self.state[p]

                # 进行状态初始化
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                # 获取当前参数的动量 m 和二阶矩 v
                m = state["m"]
                v = state["v"]
                
                # 更新时间步 t
                state["t"] += 1
                t = state["t"]

                # 更新动量 m 和二阶矩 v
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction （偏置修正），用于补偿动量和二阶矩在初始阶段的偏差。
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 参数更新的时候不要构建计算图，因为我们不需要对参数更新进行反向传播。
                with torch.no_grad():
                    # AdamW 的更新规则是
                    # p = p - lr_t * (m / (sqrt(v) + eps) + weight_decay * p)
                    p.addcdiv_(m, torch.sqrt(v).add(eps), value=-lr_t)
                    p.add_(p, alpha=-weight_decay * lr)
        return loss