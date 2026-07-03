"""
Generic model trainer.
"""

from pathlib import Path

import joblib

import torch


class Trainer:

    def __init__(

        self,

        model,

        output_dir="models",

    ):

        self.model = model

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            exist_ok=True

        )

    def save(self):

        filename = (

            self.output_dir

            / f"{self.model.name}"

        )

        if hasattr(

            self.model,

            "model",

        ):

            torch.save(

                self.model.model.state_dict(),

                filename.with_suffix(".pt"),

            )

        else:

            joblib.dump(

                self.model,

                filename.with_suffix(".joblib"),

            )
