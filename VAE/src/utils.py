# src/utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_float_result(image, axs):
    for i, img in enumerate(image):
        axs[i // 4, i % 4].imshow(img)
        axs[i // 4, i % 4].axis('off')
    return axs

def visualize_binary_result(image, output_path, row=4, col=4):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for i, img in enumerate(image):
        axs[i // 4, i % 4].imshow(img, cmap='binary')
        axs[i // 4, i % 4].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"visualization.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def visualize_latent_space(latents, labels, ax):
    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()
    ax.scatter(latents[:, 0], latents[:, 1], c=labels, s=10, cmap='hsv')
    return ax
