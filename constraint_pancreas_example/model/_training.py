from typing import Optional, Union

import numpy as np

from scvi.dataloaders import DataSplitter
from scvi.train import TrainRunner

from constraint_pancreas_example.train import TrainingPlanMixin


class TrainingMixin:
    """General purpose unsupervised train method."""

    # TODO could make custom Trainer (in a custom TrainRunner) to have in init params for early stopping
    #  "loss" rather than "elbo" components in available param specifications - for now just use
    #  a loss that is against the param specification

    # TODO run and log val before training - already tried some solutions by calling trainer.validate before
    #  fit and num_sanity_val_steps (designed not to log)
    train_runner_cls = TrainRunner
    training_plan_cls = TrainingPlanMixin
    data_splitter_cls = DataSplitter

    def train(
            self,
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            plan_kwargs: Optional[dict] = None,
            **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = self.data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = self.training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs['early_stopping_monitor'] = 'loss'
        runner = self.train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
