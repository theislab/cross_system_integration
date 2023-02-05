from typing import Optional
import numpy as np

import scvi.dataloaders
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.model._utils import parse_use_gpu_arg


class DataSplitter(scvi.dataloaders.DataSplitter):
    """
    Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
            self,
            adata_manager: AnnDataManager,
            indices: np.array,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(
            adata_manager=adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
            **kwargs
        )

        self.indices = indices

        # Redo splitting size estimation based on indices
        self.n_train, self.n_val = validate_data_split(
            self.indices.shape[0], self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n_train = self.n_train
        n_val = self.n_val
        random_state = np.random.RandomState(seed=settings.seed)
        permutation = random_state.permutation(self.indices)
        self.val_idx = permutation[:n_val]
        self.train_idx = permutation[n_val: (n_val + n_train)]
        self.test_idx = permutation[(n_val + n_train):]

        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )
