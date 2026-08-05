import torch
import matplotlib.pyplot as plt
import numpy as np


def run_visual_test(model, dataset, device, num_samples=5):
    """
    Выполняет инференс на нескольких случайных примерах и визуализирует результат.
    """
    model.eval()
    print(f"\n--- Starting Visual Inference on {num_samples} samples ---")

    # Выбираем случайные индексы
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Подготовка фигуры matplotlib
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 1. Получаем данные для модели (через __getitem__)
            sample_data = dataset[idx]
            events = sample_data["events"].unsqueeze(0).to(device)  # [1, L]
            ts = sample_data["ts"].unsqueeze(0).to(device)  # [1, L, 1]
            mask = sample_data["mask"].unsqueeze(0).to(device)  # [1, L]
            true_label = sample_data["label"].item()

            # 2. Инференс
            logits = model(events, ts, mask)
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred_label].item()

            # 3. Получаем картинку для визуализации
            image = dataset.get_raw_image(idx)

            # 4. Отрисовка
            ax = axes[i]
            ax.imshow(image, cmap='gray')
            color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f"P: {pred_label} ({confidence:.2f})\nT: {true_label}", color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    print("--- Visual Inference Finished ---\n")