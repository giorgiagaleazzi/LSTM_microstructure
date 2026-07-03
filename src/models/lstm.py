"""
Long Short-Term Memory forecasting model.
"""

from __future__ import annotations

import torch

from torch import nn

from .base import ForecastModel


class LSTMNet(nn.Module):

    def __init__(

        self,

        input_size,

        hidden_size=64,

        layers=2,

        dropout=0.2

    ):

        super().__init__()

        self.hidden = hidden_size

        self.layers = layers

        self.lstm = nn.LSTM(

            input_size,

            hidden_size,

            num_layers=layers,

            batch_first=True,

            dropout=dropout

        )

        self.output = nn.Linear(

            hidden_size,

            1

        )

    def forward(

        self,

        x

    ):

        h, _ = self.lstm(x)

        return self.output(

            h[:, -1]

        )
