from torch_geometric.data import Dataset

# from diffusion_hopping.data.dataset.datamodule import ProteinLigandComplexDataModule
from diffusion_hopping.data.dataset.datamodule_consistency import ConsistencyProteinLigandComplexDataModule

from diffusion_hopping.data.dataset.pdbbind.dataset import PDBBindDataset


class PDBBindDataModule(ConsistencyProteinLigandComplexDataModule):
    def __init__(
        self,
        root: str,
        pre_transform=None,
        pre_filter=None,
        batch_size=32,
        test_batch_size=None,
        val_batch_size=None,
        shuffle=False,
        overfit_item=False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            val_batch_size=val_batch_size,
            shuffle=shuffle,
            overfit_item=overfit_item,
        )
        self.root = root
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.save_hyperparameters()

    def dataset_from_split(self, split: str) -> Dataset:
        return PDBBindDataset(
            self.root,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            split=split,
        )
