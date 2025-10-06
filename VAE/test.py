# test.py
import yaml
import torch
from src.model import VarationalAutoencoder
from src.utils import visualize_samples

def main():
    with open("./configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_channel = config["image_size"] * config["image_size"]
    model = VarationalAutoencoder(
        image_size=config["image_size"],
        input_channel=input_channel,
        device=device,
        latent_dim=config["latent_dim"]
    ).to(device)

    ckpt_path = f"{config['paths']['ckpt_path']}/model_epoch_100.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"✅ 成功加载模型：{ckpt_path}")

    with torch.no_grad():
        samples = model.sample(16)
        visualize_samples(samples, epoch=999, save_dir=config["paths"]["visualization_path"], image_size=config["image_size"])

if __name__ == "__main__":
    main()
