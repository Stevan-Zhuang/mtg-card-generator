from config import get_config
from text_model import unicode_filter, string_tensor

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

import numpy as np
import pandas as pd
import string
import random

from typing import Optional, List, Dict, Tuple
import argparse

class MTGTextDataset(Dataset):
    """
    Helper dataset that stores a list of strings and retrieves them as
    tensors.
    """
    def __init__(self, texts: List[str], vocab: str, char2idx: Dict[str, int],
                 max_length: int, replace: Dict[str, str]) -> None:
        """Construct dataset and clean string data."""
        self.vocab = vocab
        self.char2idx = char2idx
        self.max_length  = max_length

        self.texts = [unicode_filter(text, self.vocab, replace)
                      for text in texts]

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve tensor version of string at index."""
        line = self.texts[idx].ljust(self.max_length + 1, "$")
        x = string_tensor(line[:-1], self.char2idx)
        y = string_tensor(line[1:], self.char2idx)
        return x, y

class MTGTextDataModule(pl.LightningDataModule):
    """Pytorch Lightning datamodule for name and text datasets."""
    def __init__(self, dataset: str, config: argparse.Namespace) -> None:
        """Construct datamodule."""
        super(MTGTextDataModule, self).__init__()
        self.dataset = dataset
        self.config = config

    def change_dataset(self, dataset: str) -> None:
        """Switch selected dataset to name or text."""
        self.dataset = dataset
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Read, build, and split datasets."""
        card_names = []
        card_texts = []
        card_flavors = []

        json_path1 = "ModernCards.json"
        cards = pd.read_json(f"{self.config.text_data_dir}/{json_path1}")
        cards.loc["text"] = cards.loc["text"].replace(np.nan, "")

        for card in cards:
            card_names.append(cards[card]["name"])
            if cards[card]["text"] != "":
                card_texts.append(cards[card]["text"])

        json_path2 = "scryfall-default-cards.json"
        cards = pd.read_json(f"{self.config.text_data_dir}/{json_path2}")

        card_flavors.extend(cards["flavor_text"].dropna().tolist())

        name_dataset = MTGTextDataset(
            card_names, self.config.name_vocab,
            self.config.name_char2idx, self.config.name_max_length,
            self.config.accent2normal
        )
        text_dataset = MTGTextDataset(
            card_texts, self.config.text_vocab,
            self.config.text_char2idx, self.config.text_max_length,
            self.config.accent2normal
        )
        flavor_dataset = MTGTextDataset(
            card_flavors, self.config.flavor_vocab,
            self.config.flavor_char2idx, self.config.flavor_max_length,
            self.config.accent2normal
        )

        name_train, name_valid = random_split(name_dataset,
            [round(len(name_dataset) * 0.8), round(len(name_dataset) * 0.2)]
        )
        text_train, text_valid = random_split(text_dataset,
            [round(len(text_dataset) * 0.8), round(len(text_dataset) * 0.2)]
        )
        flavor_train, flavor_valid = random_split(flavor_dataset,
            [round(len(flavor_dataset) * 0.8), round(len(flavor_dataset) * 0.2)]
        )

        self.datasets = {
            "name_train": name_train, "name_val": name_valid,
            "text_train": text_train, "text_val": text_valid,
            "flavor_train": flavor_train, "flavor_val": flavor_valid
        }

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(self.datasets[f"{self.dataset}_train"],
                          batch_size=self.config.text_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(self.datasets[f"{self.dataset}_val"],
                          batch_size=self.config.text_batch_size, shuffle=False)

class RNNModel(nn.Module):
    """Recurrent Neural Network."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 n_layers: int) -> None:
        """Construct model."""
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Produce model output from given input."""
        batch_size = x.size(0)
        encoded = self.encoder(x)
        y_pred, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        y_pred = self.decoder(y_pred.view(batch_size, -1))
        return y_pred, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize the hidden state of the RNN."""
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

class MTGTextModel(pl.LightningModule):
    """The main machine learning module."""
    def __init__(self, dataset: str, config: argparse.Namespace) -> None:
        """Construct Lightning Module."""
        super(MTGTextModel, self).__init__()
        self.save_hyperparameters()

        self.net = RNNModel(
            getattr(self.hparams.config, f"{self.hparams.dataset}_vocab_size"),
            self.hparams.config.text_hidden_size,
            getattr(self.hparams.config, f"{self.hparams.dataset}_vocab_size"),
            self.hparams.config.text_n_layers
        )

    def build_trainer(self) -> pl.Trainer:
        """Build model trainer."""
        return pl.Trainer(
            gpus=1 if self.hparams.config.text_gpu else 0,
            max_epochs=self.hparams.config.text_n_epochs,
            callbacks=[ModelCheckpoint("val_loss"),
                       EarlyStopping(
                        "val_loss", patience=self.hparams.config.text_patience)]
        )

    def forward(
        self, start_str: Optional[str] = None,
        predict_len: Optional[int] = 1000, temperature: Optional[float] = 0.8
        ) -> str:
        """Produce a sample of generated text."""
        if start_str is None:
            if self.hparams.dataset == "name" or self.hparams.dataset == "text":
                start_str = random.choice(string.ascii_uppercase)
            if self.hparams.dataset == "flavor":
                start_str = "\""

        hidden = self.net.init_hidden(1)
        x = string_tensor(
            start_str, getattr(self.hparams.config,
                               f"{self.hparams.dataset}_char2idx")
        ).unsqueeze(0)

        for idx in range(len(start_str) - 1):
            _, hidden = self.net(x[:, idx], hidden)
        x = x[:, -1]
        
        for idx in range(predict_len):
            y_pred, hidden = self.net(x, hidden)
            
            y_pred_distrib = y_pred.data.view(-1).div(temperature).exp()
            top_idx = torch.multinomial(y_pred_distrib, 1)[0]
            #top_idx = y_pred.argmax()

            next_char = getattr(self.hparams.config,
                                f"{self.hparams.dataset}_vocab")[top_idx]
            start_str += next_char
            x = string_tensor(
                next_char, getattr(self.hparams.config,
                                   f"{self.hparams.dataset}_char2idx")
            ).unsqueeze(0)

        return start_str.replace(
            getattr(self.hparams.config,
                    f"{self.hparams.dataset}_vocab")[-1], ""
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Calculate loss from a single training batch."""
        x, y = batch
        batch_size = x.size(0)
        string_size = x.size(1)
        hidden = self.net.init_hidden(batch_size)
        if self.hparams.config.text_gpu:
            hidden = hidden.cuda()

        loss = 0
        for idx in range(string_size):
            y_pred, hidden = self.net(x[:, idx], hidden)
            loss += F.cross_entropy(y_pred.view(batch_size, -1), y[:, idx])

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """Calculate loss from a single validation batch."""
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self) -> Adam:
        """Return optimizers for model."""
        return Adam(self.parameters(), lr=self.hparams.config.text_lr)

def run_text(config: argparse.Namespace) -> None:
    """Train the text models."""
    seed_everything(6)
    config = get_config()

    mtg_dm = MTGTextDataModule("name", config)
    name_model = MTGTextModel("name", config)
    name_trainer = name_model.build_trainer()
    name_trainer.fit(name_model, datamodule=mtg_dm)

    for _ in range(10):
        print(name_model())

    mtg_dm.change_dataset("text")
    text_model = MTGTextModel("text", config)

    text_trainer = text_model.build_trainer()
    text_trainer.fit(text_model, datamodule=mtg_dm)

    for _ in range(10):
        print(text_model())

    mtg_dm.change_dataset("flavor")
    flavor_model = MTGTextModel("flavor", config)
    flavor_trainer = flavor_model.build_trainer()
    flavor_trainer.fit(flavor_model, datamodule=mtg_dm)

    for _ in range(10):
        print(flavor_model())
