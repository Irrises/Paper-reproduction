# train.py
import yaml
import torch
from src.model import VarationalAutoencoder
from src.dataset import CustomizedDataset
from src.trainer import Trainer

def main():
    with open("./configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CustomizedDataset(root="./dataset")
    loaders = dataset.get_loaders(config["batch_size"])

    input_channel = config["image_size"] * config["image_size"]
    model = VarationalAutoencoder(
        image_size=config["image_size"],
        input_channel=input_channel,
        device=device,
        latent_dim=config["latent_dim"]
    ).to(device)

    trainer = Trainer(model, loaders, config)
    trainer.train()

if __name__ == "__main__":
    main()
