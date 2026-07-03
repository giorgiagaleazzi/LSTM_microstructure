"""
Abstract forecasting model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForecastModel(ABC):
    """
    Base class for every forecasting model.
    """

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):

        ...

    def fit_predict(self, X_train, y_train, X_test):

        self.fit(X_train, y_train)

        return self.predict(X_test)

    @property
    def name(self):

        return self.__class__.__name__
