Training:
- Training -> TrainingMixin - train method: DataSplitter, TrainingPlan, TrainRunner
- TrainRunner: fit the model via Trainer, eval, and update history
- Trainer: Just wraps the pytorch Trainer which calls fit
- TrainingPlan: optimisers, steps (train/val) incl passing loss params, stores losses/metrics

Data:
- AnnDataLoader is torch DataLoader; samples from AnnTorchDataset - mapping dataset (access elements by key via __getitem__)


