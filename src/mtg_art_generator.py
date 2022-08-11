from config import get_config
from art_model import demo_single, demo_grid

from pl_bolts.models.gans import DCGAN
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from typing import Optional
import argparse

class MTGImageDataModule(pl.LightningDataModule):
    """Pytorch Lightning datamodule for image datasets."""
    def __init__(self, config: argparse.Namespace) -> None:
        """Construct datamodule."""
        super(MTGImageDataModule, self).__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None) -> None:
        """Read and preprocess dataset images."""
        pipeline = T.Compose([
            lambda image: TF.crop(image, 23, 28, 90, 90),
            T.Resize(self.config.art_image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dataset = ImageFolder(
            self.config.art_data_dir, transform=pipeline
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(self.train_dataset,
                          batch_size=self.config.art_batch_size, shuffle=True)

def run_art(config: argparse.Namespace) -> None:
    """Train the art model."""
    seed_everything(6)
    config = get_config()

    mtg_dm = MTGImageDataModule(config)
    model = DCGAN(image_channels=config.art_n_channels)

    trainer = pl.Trainer(
        gpus=1 if config.art_gpu else 0,
        max_epochs=config.art_n_epochs,
    )

    trainer.fit(model, datamodule=mtg_dm)

    demo_single(model)
    demo_grid(model)