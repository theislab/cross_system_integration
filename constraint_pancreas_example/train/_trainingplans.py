from inspect import getfullargspec
from typing import Optional, Union, Dict

import torch
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from scvi._compat import Literal
from scvi.module.base import BaseModuleClass

# TODO could make new metric class to not be called elbo metric as used for other metrics as well
from scvi.train._metrics import ElboMetric

from constraint_pancreas_example.module._loss_recorder import LossRecorder


class WeightScaling:

    def __init__(self,
                 weight_start: float,
                 weight_end: float,
                 point_start: int,
                 point_end: int,
                 update_on: Literal['epoch', 'step'],
                 ):
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.point_start = point_start
        self.point_end = point_end
        if update_on not in ['step', 'epoch']:
            raise ValueError('update_on not recognized')
        self.update_on = update_on

        weight_diff = self.weight_end - self.weight_start
        n_points = self.point_end - self.point_start
        self.slope = weight_diff / n_points

        if self.weight(epoch=self.point_start, step=self.point_start) < 0 or \
                self.weight(epoch=self.point_end, step=self.point_end) < 0:
            raise ValueError('Specified weight scaling would lead to negative weights')

    def weight(
            self,
            epoch: int,
            step: int,
    ) -> float:
        """
        Computes the kl weight for the current step/epoch depending on which update type was set in init

        Parameters
        ----------
        epoch
            Current epoch.
        step
            Current step.
        """

        if self.update_on == 'epoch':
            point = epoch
        elif self.update_on == 'step':
            point = step
        else:
            pass

        if point < self.point_start:
            return self.weight_start
        elif point > self.point_end:
            return self.weight_end
        else:
            return self.slope * (point - self.point_start) + self.weight_start


class TrainingPlanMixin(TrainingPlan):
    """
    Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.
    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/model_user_guide`.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    lr

        Learning rate used for optimization.
    weight_decay
        Weight decay used in optimizatoin.
    eps
        eps used for optimization.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`).
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed.
    loss_weights:
        Specifies how losses should be weighted and in which part of the training
        Dict with keys being loss names and values being loss weights.
        Loss weights can be floats for constant weight or WeightScaling object
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        Loss weights should not be passed here and are handled via loss_weights param.
    """

    def __init__(
            self,
            module: BaseModuleClass,
            lr: float = 1e-3,
            weight_decay: float = 1e-6,
            eps: float = 0.01,
            optimizer: Literal["Adam", "AdamW"] = "Adam",
            reduce_lr_on_plateau: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_threshold_mode: str = 'rel',
            lr_scheduler_metric: Literal[
                "loss"
            ] = "loss",
            lr_min: float = 0,
            loss_weights: Union[None, Dict[str, Union[float, WeightScaling]]] = None,
            **loss_kwargs,
    ):
        super(TrainingPlan, self).__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_threshold_mode = lr_threshold_mode
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs

        self._n_obs_training = None
        self._n_obs_validation = None

        # automatic handling of loss component weights
        if loss_weights is None:
            loss_weights = {}
        self.loss_weights = loss_weights

        # Ensure that all passed loss weight specifications are in available loss params
        # Also update loss kwargs based on specified weights
        self._loss_args = getfullargspec(self.module.loss)[0]
        for loss, weight in loss_weights.items():
            if loss not in self._loss_args:
                raise ValueError(f'Loss {loss} for which a weight was specified is not in loss parameters')
            self.loss_kwargs.update({loss: self.compute_loss_weight(weight=weight)})

        self.initialize_train_metrics()
        self.initialize_val_metrics()

    def compute_loss_weight(self, weight):
        if isinstance(weight, float):
            return weight
        elif isinstance(weight, int):
            return float(weight)
        elif isinstance(weight, WeightScaling):
            return weight.weight(epoch=self.current_epoch, step=self.global_step)

    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: Optional[int] = None):
        """Initialize ELBO metric and the metric collection."""
        loss = ElboMetric("loss", mode, "obs")
        collection = MetricCollection(
            {metric.name: metric for metric in [loss]}
        )
        return loss, collection

    def initialize_train_metrics(self):
        """
        Initialize train related metrics.
        TODO could add other metrics
        """
        (
            self.loss_train,
            self.train_metrics,
        ) = self._create_elbo_metric_components(
            mode="train", n_total=self.n_obs_training
        )
        # Other metrics were in scvi-tools not reset
        self.loss_train.reset()

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        (
            self.loss_val,
            self.val_metrics,
        ) = self._create_elbo_metric_components(
            mode="validation", n_total=self.n_obs_validation
        )
        self.loss_val.reset()

    @torch.no_grad()
    def compute_and_log_metrics(
            self,
            loss_recorder: LossRecorder,
            metrics: MetricCollection,
            mode: str,
            metrics_eval:Optional[dict]=None,
    ):
        """
        Computes and logs metrics.

        Parameters
        ----------
        loss_recorder
            LossRecorder object from scvi-tools module
        metric_attr_name
            The name of the torch metric object to use
        mode
            Postfix string to add to the metric name of
            extra metrics
        metrics_eval
            Evaluation metrics given as dict name:metric_value
        """
        n_obs_minibatch = loss_recorder.n_obs
        loss_sum = loss_recorder.loss_sum

        # use the torchmetric object
        metrics.update(
            loss=loss_sum,
            n_obs_minibatch=n_obs_minibatch,
        )
        # pytorch lightning handles everything with the torchmetric object
        # TODO in trainer init can set how often per epoch the val is checked with val_check_interval
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=n_obs_minibatch,
        )

        # accumulate extra metrics passed to loss recorder
        for extra_metric in loss_recorder.extra_metric_attrs:
            met = getattr(loss_recorder, extra_metric)
            if isinstance(met, torch.Tensor):
                if met.shape != torch.Size([]):
                    raise ValueError("Extra tracked metrics should be 0-d tensors.")
                met = met.detach()
            self.log(
                f"{extra_metric}_{mode}",
                met,
                on_step=False,
                on_epoch=True,
                batch_size=n_obs_minibatch,
            )
        # accumulate extra eval metrics
        if metrics_eval is not None:
            for extra_metric, met in metrics_eval.items():
                if isinstance(met, torch.Tensor):
                    if met.shape != torch.Size([]):
                        raise ValueError("Extra tracked metrics should be 0-d tensors.")
                    met = met.detach()
                self.log(
                    f"{extra_metric}_{mode}_eval",
                    met,
                    on_step=False,
                    on_epoch=True,
                    batch_size=n_obs_minibatch,
                )

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        for loss, weight in self.loss_weights.items():
            self.loss_kwargs.update({loss: self.compute_loss_weight(weight=weight)})
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return torch.mean(scvi_loss.loss)

    def validation_step(self, batch, batch_idx):
        # loss kwargs here contains `n_obs` equal to n_training_obs
        # so when relevant, the actual loss value is rescaled to number
        # of training examples
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        metrics_eval = self.module.eval_metrics()
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation", metrics_eval=metrics_eval)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        if self.optimizer_name == "Adam":
            optim_cls = torch.optim.Adam
        elif self.optimizer_name == "AdamW":
            optim_cls = torch.optim.AdamW
        else:
            raise ValueError("Optimizer not understood.")
        optimizer = optim_cls(
            params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode=self.lr_threshold_mode,
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": scheduler,
                    "monitor": self.lr_scheduler_metric,
                },
            )
        return config
