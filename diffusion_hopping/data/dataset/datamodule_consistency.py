import os
from abc import abstractmethod

from pytorch_lightning import LightningDataModule
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import numpy as np

class ConsistencyProteinLigandComplexDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        test_batch_size=None,
        val_batch_size=None,
        shuffle=True,
        overfit_item=False,
        num_workers= 16
    ) -> None:
        super().__init__()
        self.predict_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if test_batch_size else batch_size
        self.val_batch_size = val_batch_size if val_batch_size else test_batch_size
        self.shuffle = shuffle
        self.overfit_item = overfit_item
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # Load individual datasets
            self.train_dataset = self.dataset_from_split("train")
            self.val_dataset = self.dataset_from_split("val")
            self.test_dataset = self.dataset_from_split("test")

            
            # Concatenate train, val, and test datasets for training
            # self.train_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
            
            # Optionally, you can still load the val_dataset separately if needed for validation during training
            # self.val_dataset = val_dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset_from_split("predict")
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def dataloader(dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    @abstractmethod
    def dataset_from_split(self, split: str) -> Dataset:
        ...

    def train_dataloader(self, num_workers=1):
        if self.overfit_item:
            train_dataset = [
                self.train_dataset[0] for _ in range(len(self.train_dataset))
            ]
        else:
            train_dataset = self.train_dataset
        return self.dataloader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=num_workers
        )

    def val_dataloader(self):
        if self.overfit_item:
            val_dataset = [self.train_dataset[0] for _ in range(self.batch_size)]
        else:
            val_dataset = self.val_dataset
        return self.dataloader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers
        )

    # def val_dataloader(self):
    #     if self.overfit_item:
    #         val_dataset = [self.train_dataset[0] for _ in range(self.batch_size)]
    #     else:
    #         # Assuming self.val_dataset is your full validation dataset
    #         full_val_dataset = self.val_dataset
            
    #         # Determine the size of the subset you want to validate on
    #         # For example, let's say you want to validate on 20% of the full dataset
    #         subset_size = int(0.2 * len(full_val_dataset))  # Adjust 0.2 to change the fraction
            
    #         # Generate random indices for the subset
    #         indices = np.random.choice(len(full_val_dataset), size=subset_size, replace=False)
            
    #         # Create a subset dataset with the randomly chosen indices
    #         val_dataset = Subset(full_val_dataset, indices)

    #     # Create and return the DataLoader for the validation dataset (or subset)
    #     return DataLoader(
    #         val_dataset,
    #         batch_size=self.val_batch_size,
    #         shuffle=False,  # Typically, you don't shuffle the validation data
    #         pin_memory=True,
    #         num_workers=self.num_workers
    #     )

    def test_dataloader(self):
        return self.dataloader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return self.dataloader(
            self.predict_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def get_train_smiles(self):
        if self.train_dataset is None:
            self.setup("fit")
        if self.overfit_item:
            return [Chem.MolToSmiles(self.train_dataset[0].ligand.ref)]
        return [Chem.MolToSmiles(item["ligand"].ref) for item in self.train_dataset]
