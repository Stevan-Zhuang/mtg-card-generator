import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import string
import random

from typing import Optional, List, Dict, Tuple

def unicode_filter(name: str, vocab:  str, replace: Dict[str, str]) -> str:
    """
    Filter out all characters that are not part of vocab while removing
    unicode characters.
    """
    normal_name = "".join(replace.get(char, char) for char in name)
    return "".join(char for char in normal_name
                   if char in vocab).strip()

def string_tensor(string: str, char2idx: Dict[str, int]) -> torch.Tensor:
    """
    Produce a tensor from a string with all characters mapped to an integer
    index.
    """
    return torch.from_numpy(np.array(
        [char2idx[char] for char in string],
    )).long()

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
            self.hparams.config.hidden_size,
            getattr(self.hparams.config, f"{self.hparams.dataset}_vocab_size"),
            self.hparams.config.n_layers
        )

    def build_trainer(self) -> pl.Trainer:
        """Build model trainer."""
        return pl.Trainer(
            gpus=1 if self.hparams.config.gpu else 0,
            max_epochs=self.hparams.config.n_epochs,
            callbacks=[ModelCheckpoint("val_loss"),
                       EarlyStopping("val_loss", patience=self.config.patience)]
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

            next_char = getattr(self.hparams.config, f"{self.hparams.dataset}_vocab")[top_idx]
            start_str += next_char
            x = string_tensor(
                next_char, getattr(self.hparams.config, f"{self.hparams.dataset}_char2idx")
            ).unsqueeze(0)

        return start_str.replace(
            getattr(self.hparams.config, f"{self.hparams.dataset}_vocab")[-1], ""
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Calculate loss from a single training batch."""
        x, y = batch
        batch_size = x.size(0)
        string_size = x.size(1)
        hidden = self.net.init_hidden(batch_size)
        if self.hparams.config.gpu:
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
        return Adam(self.parameters(), lr=self.hparams.config.lr)