import time

from utils.console_colors import YELLOW, CLEAR, RED
from training.train_pipeline.train_utils import run_epoch

def train_pipeline(model, train_loader, test_loader, optimizer, scheduler,
                   curriculum_manager, device, epochs, writer=None, log_interval=5):

    print(f"Learnings start. Total max steps:: {epochs}\n")
    start_time = time.time()

    for epoch in range(epochs):
        if curriculum_manager is not None:
            curriculum_manager.step(epoch)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            is_training=True
        )

        val_metrics = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            is_training=False
        )

        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        elapsed_time = time.time() - start_time
        timer_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        if epoch%log_interval == 0:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"| Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} "
                  f"| Test Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                  f"| [{timer_str}] ")

        if train_metrics['loss'] < 0.5 and val_metrics['loss'] > 2.0:
            print(f"[Epoch {epoch}]" + RED + " Warning: Possible Overfit" + CLEAR)
    print(YELLOW + "finished" + CLEAR)