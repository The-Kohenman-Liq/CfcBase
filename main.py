import torch

from models.CfcModel import CfcMemoryModel
from training.curriculum_manager import CurriculumManager
from training.train_pipeline.train import run_epoch
from training.train_pipeline.train_utils import make_optimizer_and_scheduler
from utils.config_loader import ProjectConfig

TRAIN_LEN = 0
LOG_INTERVAL = 5

def main():
    cfg = ProjectConfig()
    print("Configuration loaded")

#===================================DATA COMPONENT INITIALISATION=========================#
    # извлечение конфигов датасета/расписания. Инициализация всяких dataset generator и тп.
    model_config = cfg.model_params

#==================================MODEL INITIALISATION====================================#
#==========================================================================================#
    embedding_dim = model_config.embedding_dim
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    layers = model_config.layers

    model = CfcMemoryModel(vocab_size=..., embedding_dim=embedding_dim,
                           device=device, h_layers_param=layers)


    model.to(device)
    print(f"Device selected:: {device}\n")

#=========================================ЗАЛУПА==========================================#
    optimizer, scheduler = make_optimizer_and_scheduler(model)

#===============================CURRICULUM MANAGER INITIALISATION=========================#
    curriculum = CurriculumManager(cfg.curriculum)
    #
    # def update_gen_context(name, era_cfg):
    #     gen.set_era_context(
    #         range_val=era_cfg['range'],
    #         num_assoc=era_cfg['num_assoc_commands'],
    #         weight=era_cfg['weight']
    #     )
    #
    # def update_scheduler_on_era_change(name, era_cfg):
    #     nonlocal scheduler
    #
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode="min", patience=3, factor=0.25, cooldown=4
    #     )
    #
    # def log_era_change_to_tb(name, era_cfg):
    #     writer.add_text("Curriculum/Era_Changes", f"Переход на эру: {name} (Параметры: {era_cfg})")
    # curriculum.register_listener(
    #     lambda name, era_cfg: setattr(train_dataset, 'num_assoc_commands', era_cfg['num_assoc_commands'])
    # )
    # curriculum.register_listener(
    #     lambda name, era_cfg: setattr(val_dataset, 'num_assoc_commands', era_cfg['num_assoc_commands'])
    # )
    # curriculum.register_listener(update_scheduler_on_era_change)
    # curriculum.register_listener(update_gen_context)
    # curriculum.register_listener(log_era_change_to_tb)


    if curriculum.eras:
        first_era_name, first_era_cfg = curriculum.eras[0]
        # Имитируем первый переход, чтобы проинициализировать объекты
        curriculum.switch_to_era(0, first_era_name, first_era_cfg)

    # try:
    #     sample_batch = next(iter(train_loader))
    #     writer.add_graph(model, sample_batch["input_ids"].to(device))
    # except Exception as e:
    #     print(f"\033[31mНе удалось записать граф модели: {e}\033[0m")

    total_max_steps = TRAIN_LEN
    print(f"Learnings start. Total max steps:: {total_max_steps}\n")

    for epoch in range(total_max_steps):
        curriculum.step(epoch)
        train_metrics = run_epoch(...)
        val_metrics = run_epoch(...)

        if epoch % LOG_INTERVAL == 0:
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {train_metrics['loss']:.4f}| Val Loss: {val_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f}  "
                  f"Era: {curriculum.current_era_name}")

        if train_metrics['loss'] < 0.5 and val_metrics['loss'] > 2.0:
            print(f"[Epoch {epoch}]\033[31m Warning: Possible Overfit\033[0m")


if __name__ == '__main__':
    main()