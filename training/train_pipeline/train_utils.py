import torch
from torch import optim
import torch.nn.functional as F


def make_optimizer_and_scheduler(model, lr=0.0005):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.25, cooldown=4
    )
    return optimizer, scheduler

def compute_diversity_loss(parallel_hidden_seq, eps: float = 1e-8) -> torch.Tensor:
    if not parallel_hidden_seq:
        return torch.tensor(0.0)

    device = parallel_hidden_seq[0][0].device
    total_ortho_loss = torch.tensor(0.0, device=device)

    for h_next_list in parallel_hidden_seq:
        num_blocks = len(h_next_list)
        if num_blocks < 2:
            continue

        hidden_size = h_next_list[0].size(-1)

        pair_loss = torch.tensor(0.0, device=device)
        pairs_count = 0

        for i in range(num_blocks):
            for j in range(i + 1, num_blocks):
                cos_sim = F.cosine_similarity(h_next_list[i], h_next_list[j], dim=-1, eps=eps)
                ortho_term = torch.mean(cos_sim ** 2)

                mean_abs_i = torch.mean(torch.abs(h_next_list[i]), dim=-1)
                mean_abs_j = torch.mean(torch.abs(h_next_list[j]), dim=-1)

                alive_penalty_i = torch.mean(F.relu(0.5 - mean_abs_i) ** 2)
                alive_penalty_j = torch.mean(F.relu(0.5 - mean_abs_j) ** 2)

                pair_loss = pair_loss + ortho_term + 2.0 * (alive_penalty_i + alive_penalty_j)
                pairs_count += 1

        if pairs_count > 0:
            total_ortho_loss = total_ortho_loss + (pair_loss / pairs_count)

    return total_ortho_loss / len(parallel_hidden_seq)



