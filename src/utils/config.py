"""
Configuration reader.
"""

from pathlib import Path

import yaml


class Config:

    def __init__(self, filename: str):

        self.path = Path(filename)

        with open(self.path) as f:

            self.data = yaml.safe_load(f)

    def __getitem__(self, item):

        return self.data[item]

    def get(self, item, default=None):

        return self.data.get(item, default)
