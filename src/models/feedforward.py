"""
Feedforward Neural Network

Stage one of the proposed methodology.

Its predictions are later used as
an input to the LSTM.
"""

from __future__ import annotations

import torch

from torch import nn

from .base import ForecastModel


class FeedForwardNet(nn.Module):

    def __init__(

        self,

        input_size,

        hidden=(64, 32)

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                input_size,

                hidden[0]

            ),

            nn.ReLU(),

            nn.Linear(

                hidden[0],

                hidden[1]

            ),

            nn.ReLU(),

            nn.Linear(

                hidden[1],

                1

            )

        )

    def forward(self, x):

        return self.network(x)


class FeedForwardModel(

    ForecastModel

):

    def __init__(

        self,

        lr=0.001,

        epochs=250

    ):

        self.lr = lr

        self.epochs = epochs

    def fit(

        self,

        X,

        y

    ):

        X = torch.FloatTensor(X)

        y = torch.FloatTensor(

            y.reshape(-1, 1)

        )

        self.model = FeedForwardNet(

            X.shape[1]

        )

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=self.lr

        )

        for epoch in range(

            self.epochs

        ):

            optimizer.zero_grad()

            pred = self.model(X)

            loss = loss_fn(

                pred,

                y

            )

            loss.backward()

            optimizer.step()

    def predict(

        self,

        X

    ):

        X = torch.FloatTensor(X)

        with torch.no_grad():

            return (

                self.model(X)

                .numpy()

                .flatten()

            )
