from typing import  Any, Dict

import torch
from torch import optim, nn




def make_optimizer_and_scheduler(model, lr=0.0005):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.25, cooldown=4
    )
    return optimizer, scheduler


def run_epoch(model: nn.Module,
              loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler,
              device: torch.device,
              is_training: bool = True) -> Dict[str, float]:

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            events = batch["events"].to(device)
            ts = batch["ts"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label"].to(device)

            if is_training:
                optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(False): ...
            logits = model(events, ts, mask)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if hasattr(model, 'scheduler_step'):
                    model.scheduler_step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            avg_loss =  total_loss / len(loader)
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return { "loss": avg_loss, "acc": accuracy}