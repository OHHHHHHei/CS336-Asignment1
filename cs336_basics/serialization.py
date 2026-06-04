import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int | None = None,
    path: str | None = None,
    epoch: int | None = None,
) -> None:
    if iteration is None:
        iteration = epoch
    if iteration is None:
        raise ValueError("iteration must be provided")
    if path is None:
        raise ValueError("path must be provided")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
