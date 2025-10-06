# test.py
import torch
import yaml
from src.model import VariationalAutoencoder
from src.utils import visualize_float_result
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open("./configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_channel = config['image_size'] * config['image_size']
    model = VarationalAutoencoder(
        image_size=config['image_size'],
        input_channel=input_channel,
        device=device,
        latent_dim=config['latent_dim']
    ).to(device)

    ckpt_path = f"{config['paths']['ckpt_path']}/model_epoch_100.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"✅ 成功加载模型：{ckpt_path}")

    with torch.no_grad():
        samples = model.sample(16).reshape(16, config['image_size'], config['image_size']).cpu().numpy()
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(samples, axs)
        plt.show()

if __name__ == "__main__":
    main()
