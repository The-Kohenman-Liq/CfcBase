import torch
import torch.nn.functional as F

from models.CfcModel import CfcMemoryModel


def run_epoch(model: CfcMemoryModel, loader, optimizer, scheduler,
              vocab, device, is_training=True, MAX_RATIO: float = 50,
              writer=None, epoch=0):

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_target_sum = 0

    # Контекст градиентов зависит от режима
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            ts = batch["ts"].to(device)

            if is_training:
                optimizer.zero_grad()

            #with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits = model(input_ids, ts)
            output_logits = logits[:, -1, :]

            zeros_count = (targets == 0).sum().item()
            pos_sum = (targets > 0).sum().item()
            ratio = zeros_count / pos_sum if pos_sum > 0 else 1.0
            ratio = min(ratio, MAX_RATIO)

            pos_weight = torch.full((vocab.size,), ratio, device=device)
            pos_weight[0] = 0.0

            loss = F.binary_cross_entropy_with_logits(output_logits, targets, pos_weight=pos_weight)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                model.scheduler_step()

            total_loss += loss.item()

            # Расчет Accuracy (это уже не Accuracy)
            probs = torch.sigmoid(output_logits)
            preds = (probs >= targets / 2.0).float()
            total_correct += ((preds == 1.0) & (targets > 0.0)).sum().item()
            total_target_sum += targets.sum().item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_target_sum if total_target_sum > 0 else 0.0

    if is_training and scheduler is not None:
        scheduler.step(avg_loss)
        if writer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Params/Learning_Rate", current_lr, epoch)

    if is_training and writer is not None and epoch % 10 == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    return {"loss": avg_loss, "acc": avg_acc}
