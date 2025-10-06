# src/utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_samples(sample_images, epoch, save_dir, image_size=28):
    os.makedirs(save_dir, exist_ok=True)
    sample_images = sample_images.reshape(sample_images.shape[0], image_size, image_size).cpu().numpy()
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, img in enumerate(sample_images):
        axs[i // 4, i % 4].imshow(img, cmap="gray")
        axs[i // 4, i % 4].axis("off")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.jpg")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"🖼️ 生成样本已保存至 {save_path}")

def visualize_latent_space(latents, labels, epoch, writer):
    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()
    fig, ax = plt.subplots()
    scatter = ax.scatter(latents[:, 0], latents[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter)
    writer.add_figure(f"latent_space_epoch_{epoch+1}", fig, epoch)
    plt.close(fig)
