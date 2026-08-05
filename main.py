import torch
from torch.utils.data import DataLoader

from models.CfcModel import CfcMemoryModel
from training.curriculum_manager import CurriculumManager
from training.train_pipeline.train import run_epoch, train_pipeline
from training.train_pipeline.train_utils import make_optimizer_and_scheduler
from utils.config_loader import ProjectConfig

BATCH_SIZE = 128*64
TRAIN_LEN = 180
LOG_INTERVAL = 1

def main():
    cfg = ProjectConfig()
    print("Configuration loaded")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

#===================================DATA COMPONENT INITIALISATION=========================#

    model_config = cfg.model_params

#==================================MODEL INITIALISATION====================================#
#==========================================================================================#

    embedding_dim = model_config.embedding_dim
    layers = model_config.layers

    model = CfcMemoryModel(vocab_size=10, embedding_dim=embedding_dim,
                           device=device, h_layers_param=layers)

    model.to(device)
    print(f"Device selected:: {device}\n")

#=========================================ЗАЛУПА==========================================#
    optimizer, scheduler = make_optimizer_and_scheduler(model)

#===============================CURRICULUM MANAGER INITIALISATION=========================#
    curriculum = CurriculumManager(cfg.curriculum)

    if curriculum.eras:
        first_era_name, first_era_cfg = curriculum.eras[0]
        # Имитируем первый переход, чтобы проинициализировать объекты
        curriculum.switch_to_era(0, first_era_name, first_era_cfg)

    total_max_steps = TRAIN_LEN


    train_pipeline(
        model=model,
        train_loader=...,
        test_loader=...,
        optimizer=optimizer,
        scheduler=scheduler,
        curriculum_manager=curriculum,
        device=device,
        epochs=total_max_steps,
        log_interval= LOG_INTERVAL
    )




if __name__ == '__main__':
    main()