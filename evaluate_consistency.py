import os
import torch
import torch_scatter
from diffusion_hopping.model import util as util
from torchvision.transforms import ToTensor
from typing import Iterable, Optional, Union
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from consistency.models_consistency import *
from train_consistency import *
from _util_consistency import *
import time
import functools
import itertools
import shutil
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from diffusion_hopping.analysis.build import MoleculeBuilder
from diffusion_hopping.analysis.evaluate.qvina import qvina_score
from diffusion_hopping.analysis.evaluate.util import (
    _image_with_highlighted_atoms,
    _to_smiles,
    _to_smiles_image,
    image_formatter,
    to_html,
)
from diffusion_hopping.analysis.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularSAScore,
    MolecularValidity,
)
from diffusion_hopping.analysis.transform import (
    LargestFragmentTransform,
    UniversalForceFieldTransform,
)
from rdkit.Chem import QED
import torch
import sascorer
import datetime
import re

image_to_tensor = ToTensor()

class ConsistencySamplingAndEditing_DiffHopp:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """
    def calculate_unified_score(self, qed_score, sa_raw_score):
        """ Normalize SA score and calculate the unified score as an average of QED and normalized SA. """
        sa_score_normalized = (10 - sa_raw_score) / 9
        return (qed_score + sa_score_normalized) / 2

    def postprocess_molecules(self, x_molecule_list, num_steps=40):
        best_mols = []

        for i in range(len(x_molecule_list[0])):  # Iterate over each molecule in the last set
            mol_list = [x[i] for x in x_molecule_list[-num_steps:]]
            highest_unified_score = -1
            best_mol = None 
            for mol in mol_list:
                if mol is not None and '.' not in Chem.MolToSmiles(mol):
                    qed_value = QED.qed(mol)
                    sa_raw_score = sascorer.calculateScore(mol)
                    unified_score = self.calculate_unified_score(qed_value, sa_raw_score)
                    if unified_score > highest_unified_score:
                        highest_unified_score = unified_score
                        best_mol = mol
            if best_mol:
                best_mol.SetProp("QED", str(QED.qed(best_mol)))
                best_mol.SetProp("SA_Score", str(sascorer.calculateScore(best_mol)))
                best_mol.SetProp("Unified_Score", str(highest_unified_score))
                # Optionally save the molecule to a file or further processing
                best_mols.append(best_mol)
            else:
                print("best mol not found for ", i)
                best_mol = mol_list[-1]
                best_mols.append(best_mol)

        return best_mols  # Returns the best molecule with its properties set

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
    def __call__(
        self,
        model: nn.Module,
        x_0: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        mask: Optional[Tensor] = None,
        # transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        # inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        # start_from_y: bool = False,
        # add_initial_noise: bool = True,
        # clip_denoised: bool = False,
        # verbose: bool = False,
        # **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """
        device = x_0["ligand"].x.device


        mask = model.consistency_model.get_mask(x_0)
        mean = torch_scatter.scatter_mean(
            x_0["ligand"].pos[mask],
            x_0["ligand"].batch[mask],
            dim=0,
            dim_size=x_0.num_graphs,
        )
        # 1. center and normalize
        x_0 = model.consistency_model.centered_complex(x_0, mask)
        x_0 = model.consistency_model.normalize(x_0)

        # 2. get noise and send it to timestep T(full noise)

        with torch.no_grad():
            x = x_0.clone()
            sigma = sigmas[0]
            sigma = torch.full((x["ligand"].x[mask].shape[0],), sigma, dtype=torch.float, device=device)
            # x["ligand"].x = x["ligand"].x.to(dtype=torch.float)
            # x["ligand"].pos = x["ligand"].pos.to(dtype=torch.float)
            
            x["protein"].x = x["protein"].x.to(dtype=torch.float)
            x["protein"].pos = x["protein"].pos.to(dtype=torch.float)
            x["ligand"].x[mask] = (
                # x_0["ligand"].x[mask].to(dtype=torch.float) + 
                torch.randn_like(x_0["ligand"].x[mask], device=device) 
            )

            x["ligand"].pos[mask] = (
                # x_0["ligand"].pos[mask].to(dtype=torch.float) + 
                util.centered_batch(
                torch.randn_like(x["ligand"].pos[mask], device=device),
                x["ligand"].batch[mask],
                dim_size=x.num_graphs,
            ))
            # x["ligand"].pos[mask] = (
            #     torch.randn_like(x["ligand"].pos[mask], device=device)
            # )
            # x_temp = x.clone()
            # x_temp_1 = model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x_temp.detach()), mean=mean)
            x["ligand"].x[mask], x["ligand"].pos[mask] = model_forward_wrapper(model, x, mask, sigma.unsqueeze(-1), self.sigma_data, self.sigma_min)
            # x_list = [model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x.detach()), mean=mean)]
            x_molecule_list= []
            for stp, sigma in tqdm(enumerate(sigmas[:-1]), desc='Processing',total=len(sigmas)-1):
                sigma = torch.full((x["ligand"].x[mask].shape[0],), sigma, dtype=torch.float, device=device)
                x_eps = torch.randn_like(x["ligand"].x[mask], device=device)
                # shape: (masked_nodes, num_features)
                pos_eps = util.centered_batch(
                    torch.randn_like(x["ligand"].pos[mask], device=device) ,
                    x["ligand"].batch[mask],
                    dim_size=x.num_graphs,
                )  # shape: (masked_nodes, 3)
                # pos_eps = torch.randn_like(x["ligand"].pos[mask], device=device)
                x["ligand"].x[mask] = (
                x["ligand"].x[mask].to(dtype=torch.float) + 
                pad_dims_like((sigma**2 - self.sigma_min**2) ** 0.5, x["ligand"].x[mask].to(dtype=torch.float)) * x_eps)

                x["ligand"].pos[mask] = (
                    x["ligand"].pos[mask].to(dtype=torch.float) + 
                    pad_dims_like((sigma**2 - self.sigma_min**2) ** 0.5, x["ligand"].pos[mask].to(dtype=torch.float)) * pos_eps
                )

                x["ligand"].x[mask], x["ligand"].pos[mask] = model_forward_wrapper(model, x, mask, sigma.unsqueeze(-1), self.sigma_data, self.sigma_min)

                # x["ligand"].pos[mask] = util.centered_batch(
                #     x["ligand"].pos[mask],
                #     x["ligand"].batch[mask],
                #     dim_size=x.num_graphs,
                # )
                x["ligand"].pos[mask] = util.centered_batch(
                    x["ligand"].pos[mask],
                    x["ligand"].batch[mask],
                    dim_size=x.num_graphs,
                )
                x_temp = x.clone()
                # x_list.append(model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x.detach()), mean=mean))
                x_molecule_list.append(model.molecule_builder(model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x_temp.detach()), mean=mean)))

        return x_molecule_list
        

class Evaluator(object):
    def __init__(self, path: Path, sigmas, find_best):
        self.data_module = None
        self.model = None
        self.molecule_builder = MoleculeBuilder(include_invalid=True)
        self.transforms = Compose(
            [LargestFragmentTransform(), UniversalForceFieldTransform()]
        )
        self._output = None
        self.molecular_metrics = None
        self._path = path
        self._metric_columns = []
        self._mode = None
        self.sigmas = sigmas
        self.consistency_sampling = ConsistencySamplingAndEditing_DiffHopp()
        self.config = LitConsistencyModelConfig
        self.find_best = find_best
    def reset_output(self):
        self._output = None

    def _setup_molecular_metrics(self):
        self.molecular_metrics = {
            "Novelty": MolecularNovelty(self.data_module.get_train_smiles()),
            "Validity": MolecularValidity(),
            "Connectivity": MolecularConnectivity(),
            "Lipinski": MolecularLipinski(),
            "LogP": MolecularLogP(),
            "QED": MolecularQEDValue(),
            "SAScore": MolecularSAScore(),
        }
        self._metric_columns = list(self.molecular_metrics.keys())

    def load_data_module(self, data_module):
        self.data_module = data_module
        self._setup_molecular_metrics()

    def load_model(self, model):
        self.model = model

    def generate_molecules(
        self, molecules_per_pocket=3, batch_size=32, limit_samples=None
    ):
        self._mode = "sampling"
        

        self._generate_molecules(
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
            limit_samples=limit_samples,
        )

    def generate_molecules_inpainting(
        self, molecules_per_pocket=3, batch_size=32, limit_samples=None, r=10, j=10
    ):
        self._mode = "inpainting"
        self._generate_molecules(
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
            limit_samples=limit_samples,
            inpaint_scaffold=True,
            r=r,
            j=j,
        )

    def use_ground_truth_molecules(self, limit_samples=None):
        self._mode = "ground_truth"
        self._use_ground_truth_molecules(limit_samples=limit_samples)

    def evaluate(self, transform_for_qvina=True):
        self.enrich_molecule_output()
        self.add_metrics()
        self.store_pockets()
        self.store_molecules(transform=transform_for_qvina)
        self.calculate_qvina_scores()

    def _prepare_dataframe(self, molecules_per_pocket):
        test_loader = self.data_module.test_dataloader()
        test_items = []
        for batch in test_loader:
            test_items.extend(batch.to_data_list())
        test_items, sample_nums = zip(
            *[(item, i) for item in test_items for i in range(molecules_per_pocket)]
        )
        self._output = pd.DataFrame(
            {
                "sample_num": sample_nums,
                "test_set_item": test_items,
            }
        )

        self._output["identifier"] = self._output["test_set_item"].apply(
            lambda x: x.identifier
        )
        self._output = self._output[["identifier", "sample_num", "test_set_item"]]
        self._output = self._output.sort_values(by=["identifier", "sample_num"])

    def _generate_molecules(
        self,
        molecules_per_pocket=3,
        batch_size=32,
        limit_samples=None,
        inpaint_scaffold=False,
        j=10,
        r=10,
    ):
        print("Generating molecules...")
        # sampling_sigmas = reversed(karras_schedule(
        # self.config.final_timesteps, self.config.sigma_min, self.config.sigma_max, self.config.rho, self.model.device
        #     ))
        # sampling_sigmas= reversed(sampling_sigmas)
        # sampling_sigmas[-1] += 1e-8 
        print("sampling sigmas have step: ", len(self.sigmas))
        self.model.eval()
        # self.data_module.setup(stage="test")
        self._prepare_dataframe(molecules_per_pocket=molecules_per_pocket)
        if limit_samples is not None:
            self._output = self._output.iloc[:limit_samples]
        # import pdb; pdb.set_trace()

        device_is_cpu = self.model.device == torch.device("cpu")
        self._output["molecule"],self._output["time"] = self._sample_molecules(
            self._output["test_set_item"],
            batch_size,
            inpaint_scaffold,
            j,
            r,
            multi_threading=device_is_cpu,
            sigmas = self.sigmas
        )

    def _use_ground_truth_molecules(self, limit_samples=None):
        print("Using ground truth molecules...")
        self.model.eval()
        self.data_module.setup(stage="test")
        self._prepare_dataframe(molecules_per_pocket=1)

        self._output["molecule"] = self._output["test_set_item"].apply(
            lambda x: x["ligand"].ref
        )
        if limit_samples is not None:
            self._output = self._output.iloc[:limit_samples]

    def enrich_molecule_output(self):
        print("Enriching molecule output...")
        self._output["SMILES"] = self._output.apply(_to_smiles, axis=1)
        self._output["Image"] = self._output.apply(self._to_image, axis=1)
        self._output["SMILES-Image"] = self._output.apply(_to_smiles_image, axis=1)

    def add_metrics(self):
        print("Adding metrics...")
        for metric_name, metric in self.molecular_metrics.items():
            self._output[metric_name] = self._output["molecule"].apply(
                lambda x: metric([x]).item()
            )
            
        self.add_diversity_metric()

    def add_diversity_metric(self):
        if "Diversity" not in self._metric_columns:
            self._metric_columns.append("Diversity")

        self._output["Diversity"] = self._output.groupby("identifier")[
            "molecule"
        ].transform(lambda x: self._calculate_diversity(x))

    def _calculate_diversity(self, x):
        mols = [mol for mol in x if mol is not None]
        if len(mols) == 0:
            return 0.0
        if len(mols) == 1:
            return 1.0

        rdk_fingerprints = [Chem.RDKFingerprint(mol) for mol in mols]

        tanimoto_similarities = [
            DataStructs.TanimotoSimilarity(f1, f2)
            for f1, f2 in itertools.combinations(rdk_fingerprints, 2)
        ]
        return 1 - np.mean(tanimoto_similarities)

    def store_molecules(self, transform=False):
        print("Storing molecules...")
        store_path = self._path / "data"
        self._output["molecule_path"] = self._output.apply(
            lambda row: store_path
            / row["identifier"]
            / f"sample_{row['sample_num']}.pdb"
            if row["molecule"] is not None
            else None,
            axis=1,
        )
        for i, row in tqdm(list(self._output.iterrows())):
            if row["molecule"] is None:
                continue
            self._store_molecule(row["molecule"], row["molecule_path"], transform)

    def _store_molecule(self, mol, path, transform=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if transform:
            mol = self.transforms(mol)
        Chem.MolToPDBFile(
            mol,
            str(path),
        )

    def store_pockets(self):
        print("Storing pockets...")
        store_path = self._path / "data"
        self._output["pocket_path"] = self._output.apply(
            lambda row: store_path / row["identifier"] / "pocket.pdb", axis=1
        )
        for i, row in tqdm(list(self._output.iterrows())):
            pocket_path = row["test_set_item"]["protein"].path
            row["pocket_path"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pocket_path, str(row["pocket_path"]))

    def calculate_qvina_scores(self):
        print("Calculating QVina scores...")
        scores = thread_map(
            lambda iterrows: qvina_score(iterrows[1]), list(self._output.iterrows())
        )
        self._output["QVina"] = scores
        if "QVina" not in self._metric_columns:
            self._metric_columns.append("QVina")

    def _sample_molecules(
        self,
        items,
        batch_size,
        inpaint_scaffold=False,
        r=10,
        j=10,
        multi_threading=True,
        sigmas= None
    ):
        # import pdb; pdb.set_trace()
        loader = DataLoader(list(items), batch_size=batch_size, shuffle=False)
        results_list = []
        
        if inpaint_scaffold:
            func = functools.partial(self._generate_molecule_inpaint, j=j, r=r)

        else:
            func = self._generate_molecule
        start_time = time.time() 
        if multi_threading:
            results = thread_map(func, list(loader), desc="Sampling molecules")
            for result in results:
                results_list.extend(result)
        else:
            for batch in tqdm(loader, desc="Sampling molecules"):
                results_list.extend(func(batch, self.find_best))
        end_time = time.time()
        duration = end_time - start_time
        print(f"Molecule generation completed in {duration:.2f} seconds.")
        return results_list, duration

    @torch.no_grad()
    def _generate_molecule(self, batch, find_best=True):
        batch = batch.to(self.model.device)
        sample_results = self.consistency_sampling(self.model,batch,self.sigmas)
        # molecules = self.molecule_builder(final_output)
        if find_best:
            print("finding best molecules...")
            molecules = self.consistency_sampling.postprocess_molecules(sample_results)
        else:
            molecules = sample_results[-1]
        # if not len(molecules) == len(sample_results[-1]):
        #      
        assert len(molecules) == len(sample_results[-1])
        return molecules

    @torch.no_grad()
    def _generate_molecule_inpaint(self, batch, j=10, r=10) -> List[Chem.Mol]:
        batch = batch.to(self.model.device)
        mask = batch["ligand"].scaffold_mask
        sample_results = self.model.model.inpaint(batch, mask, j=j, r=r)
        final_output = sample_results[-1]
        molecules = self.molecule_builder(final_output)
        return molecules

    def to_html(self, path):
        return to_html(
            self._output.drop(columns=["test_set_item"]),
            path,
            image_columns=["Image", "SMILES-Image"],
        )

    def to_csv(self, path):
        self._output.drop(columns=["test_set_item"]).to_csv(path)

    def to_tensor(self, path):
        torch.save((self._output, self._mode), path)

    def from_tensor(self, path):
        self._output, self._mode = torch.load(path)

    def print_summary_statistics(self):
        print(self.get_summary_string())

    def get_summary_string(self):
        summary_statistics = self.get_summary_statistics()
        summary_string = f"Summary statistics for mode {self._mode}:\n"
        for metric_name, metric_statistics in summary_statistics.items():
            summary_string += f"{metric_name}: {metric_statistics['mean']:.3f} ± {metric_statistics['std']:.3f}\n"
        summary_string += f"time: {self._output['time'].mean()}"
        
        return summary_string

    def get_summary_statistics(self):
        summary_statistics = {}
        for metric_name in self._metric_columns:
            summary_statistics[metric_name] = {
                "mean": self._output[metric_name].mean(),
                "std": self._output[metric_name].std(),
            }
        return summary_statistics

    def _get_conditional_mask(self, row, mark_scaffold=None):
        if self._mode == "ground_truth":
            return ~row["test_set_item"]["ligand"].scaffold_mask
        elif self._mode == "sampling":
            if mark_scaffold is None:
                return ~self.model.consistency_model.get_mask(row["test_set_item"])
            elif mark_scaffold:
                return ~row["test_set_item"]["ligand"].scaffold_mask
            else:
                return torch.ones_like(
                    row["test_set_item"]["ligand"].scaffold_mask
                ).bool()
        elif self._mode == "inpainting":
            return ~row["test_set_item"]["ligand"].scaffold_mask
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def _to_image(self, row):
        mask = self._get_conditional_mask(row)
        atoms_to_highlight = [item.item() for item in torch.where(mask)[0]]

        mol = row["molecule"]
        return _image_with_highlighted_atoms(mol, atoms_to_highlight)

    # def is_model_repainting_compatible(self) -> bool:
    #     return not self.model.model.condition_on_fg

    def output_best_samples(
        self,
        identifier: str,
        sample_nums: List[int],
        n=3,
        transform=True,
        mark_scaffold=True,
    ):
        output = self._output[self._output["identifier"] == identifier]
        output = output[output["sample_num"].isin(sample_nums)]
        output = output.nsmallest(n, "QVina")

        output_path = self._path / "samples" / identifier
        output_path.mkdir(parents=True, exist_ok=True)
        output["molecule_path"] = output.apply(
            lambda row: output_path / f"sample{row['sample_num']}_{self._mode}.pdb",
            axis=1,
        )
        for i, row in output.iterrows():
            self._store_molecule(
                row["molecule"], row["molecule_path"], transform=transform
            )
            qvina_score(row)

        to_html(
            output.drop(columns=["test_set_item"]),
            output_path / f"summary_{self._mode}.html",
            image_columns=["Image", "SMILES-Image"],
        )

        for i, row in output.iterrows():
            image = row["Image"]
            image.save(output_path / f"sample{row['sample_num']}_{self._mode}.png")

            smiles_image = row["SMILES-Image"]
            smiles_image.save(
                output_path / f"sample{row['sample_num']}_{self._mode}_smiles.png"
            )
        for i, row in output.iterrows():
            mol = Chem.Mol(row["molecule"])
            mask = self._get_conditional_mask(row, mark_scaffold=mark_scaffold)
            atoms_to_highlight = [item.item() for item in torch.where(mask)[0]]
            bonds_to_highlight = [
                bond.GetIdx()
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in atoms_to_highlight
                or bond.GetEndAtomIdx() in atoms_to_highlight
            ]
            from rdkit.Chem import rdCoordGen

            rdCoordGen.AddCoords(mol)

            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=atoms_to_highlight,
                highlightBonds=bonds_to_highlight,
            )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg = svg.replace("svg:", "")
            Path(
                output_path / f"sample{row['sample_num']}_{self._mode}_highlight.svg"
            ).write_text(svg)


def disable_obabel_and_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")
    openbabel.obErrorLog.SetOutputLevel(0)
    openbabel.obErrorLog.StopLogging()
    message_handler = openbabel.OBMessageHandler()
    message_handler.SetOutputLevel(0)

@dataclass
class EvalConfig:

    model_config: None
    consistency_training: ConsistencyTraining_DiffHopp
    consistency_sampling: ConsistencySamplingAndEditing_DiffHopp
    lit_cm_config: LitConsistencyModelConfig
    seed: int = 42
    ckpt_dir: str = '/data/aigen/consistency/training/checkpoints'
    resume_ckpt_path: Optional[str] = None
    device: Optional[int] = None
    check_val_every_n_epoch: Optional[int] = None
    consistency_training: None
    consistency_sampling: None
    lit_cm_config: None
    device: None
    check_val_every_n_epoch: int = 1
    wandb_logging: bool = False 

    # def __post_init__(self):
    #     # Format current date
    #     current_date = datetime.now().strftime("%Y%m%d")
    #     # Update model_ckpt_path to include final_timesteps and date
    #     self.model_ckpt_path = f"{self.ckpt_dir}/{current_date}/ver_dist_loss_gvp_{self.lit_cm_config.final_timesteps}"
    #     # max_epochs: int = 1000

def generate_molecules(
    evaluator: Evaluator,
    output_path: Path,
    mode: str = "all",
    r: int = 10,
    j: int = 10,
    limit_samples: int = None,
    molecules_per_pocket: int = 100,
    batch_size: int = 32,
):
    if (
        mode == "ground_truth"
        or mode == "all"
        # or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print("Generating ground truth molecules...")
        evaluator.use_ground_truth_molecules(limit_samples=limit_samples)
        evaluator.to_tensor(output_path / "molecules_ground_truth.pt")

    if mode == "ligand_generation" or mode == "all":
        print("Generating ligand molecules...")
        evaluator.generate_molecules(
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_ligand_generation.pt")

def evaluate_molecules(evaluator, output_path, mode="all"):

    output_str = f"Output path: {output_path}\n"
    if (
        mode == "ground_truth"
        or mode == "all"
        # or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print("Running ground truth evaluation...")
        evaluator.from_tensor(output_path / "molecules_ground_truth.pt")
        evaluator.evaluate(transform_for_qvina=False)
        evaluator.to_html(output_path / "results_ground_truth.html")
        evaluator.to_tensor(output_path / "results_ground_truth.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ground truth results: \n{evaluator.get_summary_string()}\n"

    if mode == "ligand_generation" or mode == "all":
        print("Running ligand generation evaluation...")
        evaluator.from_tensor(output_path / "molecules_ligand_generation.pt")
        evaluator.evaluate(transform_for_qvina=True)
        evaluator.to_html(output_path / "results_ligand_generation.html")
        evaluator.to_tensor(output_path / "results_ligand_generation.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ligand generation results: \n{evaluator.get_summary_string()}\n"

    output_path.joinpath("summary.txt").write_text(output_str)

def parse_checkpoint_for_final_timesteps(checkpoint_path: str) -> int:
    """
    Parse the checkpoint path to extract final_timesteps.
    Assumes path format contains 'gvp_[final_timesteps]'.
    """
    match = re.search(r"gvp_(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract final_timesteps from path: {checkpoint_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the model checkpoint file')
    parser.add_argument('--cuda_device', type=int, default=0, 
                    help='CUDA device number to use for computation')
    parser.add_argument('--molecules_per_pocket', type=int, default=10, 
                    help='molecules_per_pocket to generate')    
    parser.add_argument('--find_best', action='store_true',
                    help='Enable refinement of molecules per pocket')
    parser.add_argument('--batch_size', type=int,default=512,
                    help='generation batch')
    parser.add_argument('--mode', type=str,default='train',
                    help='training/active_learning')
    parser.add_argument('--dataset', type=str,default='pdbbind_filtered',
                    help='dataset')    
    parser.add_argument('--root_save_path', type=str,default='/data/aigen/consistency/evaluation/',
                help='root save path')   
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    limit_samples= None
    molecules_per_pocket = args.molecules_per_pocket
    batch_size = args.batch_size

    if args.mode == 'train':
        final_timesteps = parse_checkpoint_for_final_timesteps(args.checkpoint_path)

    print("final timesteps to evaluate are :  ", final_timesteps)

    model_config = SimpleNamespace(
    architecture = Architecture.GVP,
    seed=1,
    dataset_name= args.dataset,
    condition_on_fg = False,
    batch_size = 512,
    T = final_timesteps,
    lr = 1e-4,
    num_layers =6 ,
    joint_features=128,
    hidden_features=256,
    edge_cutoff=(None, 5, 5),
    )
    model_config.attention = True

    config = EvalConfig(
                model_config=model_config,
                consistency_training=ConsistencyTraining_DiffHopp(final_timesteps= final_timesteps),
                consistency_sampling=ConsistencySamplingAndEditing_DiffHopp(final_timesteps= final_timesteps),
                lit_cm_config=LitConsistencyModelConfig(
                final_timesteps= final_timesteps
                ),
                device=[args.cuda_device],
                check_val_every_n_epoch=1)

    student_model, _, teacher_model = get_consistency_models()

    model = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        config.lit_cm_config,
    )

    disable_obabel_and_rdkit_logging()

    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.student_model
    model.to(device)
    model.eval()

    data_module = get_datamodule(
    config.model_config.dataset_name, batch_size=batch_size)
    data_module.setup(stage="fit")

    sigmas = reversed(karras_schedule(
                    final_timesteps, 
                    sigma_min= 0.002, 
                    sigma_max= 80.0, 
                    rho=7, 
                    ))
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_identifier = os.path.basename(args.checkpoint_path).replace('.ckpt', '')
    output_path = f"{args.root_save_path}/{args.mode}/gvp_{final_timesteps}/{checkpoint_identifier}/find_best_{args.find_best}/{datetime_string}"

    if args.mode == 'train':
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(output_path, sigmas, args.find_best)
    evaluator.load_data_module(data_module)
    evaluator.load_model(model)
    
    generate_molecules(
        evaluator,
        output_path,
        mode="ligand_generation",
        # r=r,
        # j=j,
        limit_samples=limit_samples,
        molecules_per_pocket=molecules_per_pocket,
        batch_size=batch_size,
    )

    evaluate_molecules(evaluator, output_path, mode="ligand_generation")