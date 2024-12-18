from collections import deque
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from rdkit.Chem import Draw
from torch_geometric.data import HeteroData
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from diffusion_hopping.analysis.build import MoleculeBuilder
from diffusion_hopping.analysis.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularSAScore,
    MolecularValidity,
)
#changed to consistency
from diffusion_hopping.model.diffusion.consistency_model import ConsistencyDiffusionModel
from diffusion_hopping.model.enum import Architecture, Parametrization
from diffusion_hopping.model.consistency_estimator import ConsistencyEstimatorModel, EstimatorModel
from diffusion_hopping.model.util import skip_computation_on_oom
import os
from dataclasses import dataclass
from torch import Tensor, nn
import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from matplotlib.pyplot import xcorr
import torch_scatter
from openbabel import openbabel
from rdkit import RDLogger
from diffusion_hopping.model import util as util
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw, QED
import torch
import sascorer

image_to_tensor = ToTensor()

def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    # print("final_timesteps:",final_timesteps)
    # print("total_training_steps:",total_training_steps)
    # print("initial_timesteps:",initial_timesteps)
    # print("current_training_step:",current_training_step)
    num_timesteps = (final_timesteps + 1) ** 2 - initial_timesteps**2

    num_timesteps = current_training_step * num_timesteps / total_training_steps

    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)


    return num_timesteps + 1

def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def disable_obabel_and_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")
    openbabel.obErrorLog.SetOutputLevel(0)
    openbabel.obErrorLog.StopLogging()
    message_handler = openbabel.OBMessageHandler()
    message_handler.SetOutputLevel(0)


def model_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    mask: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)

    c_skip = pad_dims_like(c_skip, pad_dims_like(sigma, x["ligand"].x[mask].to(dtype=torch.float)))
    c_out = pad_dims_like(c_out, pad_dims_like(sigma, x["ligand"].x[mask].to(dtype=torch.float)))
    
    x_final_ligand_ligand_mask, pos_out_ligand_ligand_mask = model.consistency_model.estimator(x, sigma, mask)
    return c_skip * x['ligand'].x[mask] + c_out * x_final_ligand_ligand_mask, c_skip * x['ligand'].pos[mask] + c_out * pos_out_ligand_ligand_mask

@dataclass
class ConsistencyTrainingOutput:
    """Type of the output of the (Improved)ConsistencyTraining.__call__ method.

    Attributes
    ----------
    predicted : Tensor
        Predicted values.
    target : Tensor
        Target values.
    num_timesteps : int
        Number of timesteps at the current point in training from the timestep discretization schedule.
    sigmas : Tensor
        Standard deviations of the noise.
    loss_weights : Optional[Tensor], default=None
        Weighting for the Improved Consistency Training loss.
    """

    predicted: Tensor
    target: Tensor
    num_timesteps: int
    sigmas: Tensor
    loss_weights: Optional[Tensor] = None

class ConsistencyTraining_DiffHopp:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    """

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
        student_model: nn.Module,
        teacher_model: nn.Module,
        x_0: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs: Any,
    ) -> ConsistencyTrainingOutput:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        student_model : nn.Module
            Model that is being trained.
        teacher_model : nn.Module
            An EMA of the student model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        ConsistencyTrainingOutput
            The predicted and target values for computing the loss as well as sigmas (noise levels).
        """
        device = x_0["ligand"].x.device
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, device
        )

        mask = student_model.consistency_model.get_mask(x_0)
        x_0 = student_model.consistency_model.centered_complex(x_0, mask)
        x_0 = student_model.consistency_model.normalize(x_0)

        t = torch.randint(0, num_timesteps - 1, (x_0.num_graphs,), device=device)

        # Ensure the tensor is in floating-point format before applying torch.randn_like
        # mask = torch.ones(x_0["ligand"].num_nodes, dtype=torch.bool, device=device)
            
        x_masked_float = x_0["ligand"].x[mask].to(dtype=torch.float)  # Convert to float
        x_eps = torch.randn_like(x_masked_float, device=device)
        pos_eps = util.centered_batch(
            torch.randn_like(x_0["ligand"].pos[mask], device=device),
            x_0["ligand"].batch[mask],
            dim_size=x_0.num_graphs,
        )  # shape: (masked_nodes, 3)

        if isinstance(t, torch.Tensor) and torch.numel(t) > 1:
            t = t[x_0["ligand"].batch[mask]]  # shape: (masked_nodes, 1)
        current_sigmas = sigmas[t]
        next_sigmas = sigmas[t + 1]

        current_sigmas= current_sigmas.unsqueeze(-1)
        next_sigmas = next_sigmas.unsqueeze(-1)

        x_t = x_0.clone()
        # Convert the destination tensor to float if it's not already
        x_t["ligand"].x = x_t["ligand"].x.to(dtype=torch.float)
        x_t["ligand"].pos = x_t["ligand"].pos.to(dtype=torch.float)
        x_t["protein"].x = x_t["protein"].x.to(dtype=torch.float)
        x_t["protein"].pos = x_t["protein"].pos.to(dtype=torch.float)

        # Now perform the operation
        x_t["ligand"].x[mask] = (
            x_t["ligand"].x[mask].to(dtype=torch.float) + 
            pad_dims_like(next_sigmas, x_t["ligand"].x[mask].to(dtype=torch.float)) * x_eps
        )
        # Now perform the operation
        x_t["ligand"].pos[mask] = (
            x_t["ligand"].pos[mask].to(dtype=torch.float) + 
            pad_dims_like(next_sigmas, x_t["ligand"].pos[mask].to(dtype=torch.float)) * pos_eps
        )
        next_x, next_pos = model_forward_wrapper(student_model, x_t, mask, next_sigmas)

        with torch.no_grad():

            x_t_1 = x_0.clone()
            x_t_1["ligand"].x = x_t_1["ligand"].x.to(dtype=torch.float)
            x_t_1["ligand"].pos = x_t_1["ligand"].pos.to(dtype=torch.float)
            x_t_1["protein"].x = x_t_1["protein"].x.to(dtype=torch.float)
            x_t_1["protein"].pos = x_t_1["protein"].pos.to(dtype=torch.float)
            x_t_1["ligand"].x[mask] = (
                x_t_1["ligand"].x[mask].to(dtype=torch.float) + 
                pad_dims_like(current_sigmas, x_t_1["ligand"].x[mask].to(dtype=torch.float)) * x_eps
            )
            x_t_1["ligand"].pos[mask] = (
                x_t_1["ligand"].pos[mask].to(dtype=torch.float) + 
                pad_dims_like(current_sigmas, x_t_1["ligand"].pos[mask].to(dtype=torch.float)) * pos_eps
            )
            current_x, current_pos = model_forward_wrapper(teacher_model, x_t_1, mask, current_sigmas)

        return ConsistencyTrainingOutput(next_x, current_x, num_timesteps, sigmas), ConsistencyTrainingOutput(next_pos, current_pos, num_timesteps, sigmas)





class ConsistencySamplingAndEditing_DiffHopp:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

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
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
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
            x_list = [model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x.detach()), mean=mean)]
            
            for stp, sigma in tqdm(enumerate(sigmas[:-1]), desc='Processing',total=len(sigmas)-1):
                sigma = torch.full((x["ligand"].x[mask].shape[0],), sigma, dtype=torch.float, device=device)
                
                x_eps = torch.randn_like(x["ligand"].x[mask])
                # shape: (masked_nodes, num_features)
                pos_eps = util.centered_batch(
                    torch.randn_like(x["ligand"].pos[mask]) ,
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
                # x_temp = x.clone()
                x_list.append(model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x.detach()), mean=mean))
                # x_list.append(model.molecule_builder(x_temp))

        return x_list
        

class ConsistencyScoreSamplingAndEditing_DiffHopp:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

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
        
    def calculate_unified_score(self, qed_score, sa_raw_score):
        """ Normalize SA score and calculate the unified score as an average of QED and normalized SA. """
        sa_score_normalized = (10 - sa_raw_score) / 9
        return (qed_score + sa_score_normalized) / 2

    def postprocess_molecules(self, x_molecule_list, num_steps=40):
        best_mols = []

        for i in range(len(x_molecule_list[0])):  # Iterate over each molecule in the last set
            mol_list = [x[i] for x in x_molecule_list[-num_steps:]]
            highest_unified_score = -1
            best_mol = None  # Initialize best_mol to None

            for mol in mol_list:
                if mol is not None and '.' not in Chem.MolToSmiles(mol):
                    qed_value = QED.qed(mol)
                    sa_raw_score = sascorer.calculateScore(mol)
                    unified_score = self.calculate_unified_score(qed_value, sa_raw_score)
                    if unified_score > highest_unified_score:
                        highest_unified_score = unified_score
                        best_mol = mol

            if best_mol:  # This will now only proceed if best_mol is not None
                best_mol.SetProp("QED", str(QED.qed(best_mol)))
                best_mol.SetProp("SA_Score", str(sascorer.calculateScore(best_mol)))
                best_mol.SetProp("Unified_Score", str(highest_unified_score))
                best_mols.append(best_mol)
            else:
                print(f"Best molecule not found for index {i}; retrieving last molecule.")
                best_mol = mol_list[-1]  # Default to the last molecule if no best molecule is found
                best_mols.append(best_mol)

        return best_mols

    def __call__(
        self,
        model: nn.Module,
        x_0: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
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

            x_molecule_list = []
            x_list = []
            all_postprocessed_mols = []
            # if shorten_generation:
            #     sigmas = sigmas_new
            #     print("following shorter generation path")
            # else: 
            total_iterations = len(sigmas)  # Replace this with the actual number of items if sampling_sigmas is a generator
            for stp, sigma in tqdm(enumerate(sigmas[:-1]), desc='Processing',total=total_iterations-1):
                sigma = torch.full((x["ligand"].x[mask].shape[0],), sigma, dtype=torch.float, device=device)
                x_eps = torch.randn_like(x["ligand"].x[mask], device=device)
                # shape: (masked_nodes, num_features)
                pos_eps = util.centered_batch(
                    torch.randn_like(x["ligand"].pos[mask], device=device) ,
                    x["ligand"].batch[mask],
                    dim_size=x.num_graphs,
                )  # shape: (masked_nodes, 3)
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
                # x_temp = model.consistency_model.uncentered_complex(model.consistency_model.denormalize(x_temp.detach()), mean=mean)
                x_list.append(x_temp)
                x_molecule_list.append(model.molecule_builder(x_temp))
            
            all_postprocessed_mols.extend(self.postprocess_molecules(x_molecule_list))
            # all_final_mols.extend(x_molecule_list[-1])
        return all_postprocessed_mols

class ConsistencyDiffusionHoppingModel(pl.LightningModule):
    def __init__(
        self,
        # Diffusion parameters
        T=500,
        parametrization=Parametrization.EPS,
        # Training parameters
        lr=1e-4,
        clip_grad=False,
        condition_on_fg=False,
        # Normalization parameters
        pos_norm=1.0,
        x_norm=1.0,
        x_bias=0.0,
        # Estimator parameters
        architecture: Architecture = Architecture.GVP,
        edge_cutoff=None,
        hidden_features=256,
        joint_features=32,
        num_layers=6,
        attention=True,
        # Dataset parameters
        ligand_features=10,
        protein_features=20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.atom_features = 10
        self.c_alpha_features = 20

        model_params = dict(
            hidden_features=hidden_features,
            num_layers=num_layers,
            attention=attention,
        )

        estimator = EstimatorModel(
            ligand_features=ligand_features,
            protein_features=protein_features,
            joint_features=joint_features,
            edge_cutoff=edge_cutoff,
            architecture=architecture,
            egnn_velocity_parametrization=(parametrization == Parametrization.EPS),
            **model_params,
        )

        self.consistency_model = ConsistencyDiffusionModel(
            estimator,
            T=T,
            parametrization=parametrization,
            pos_norm=pos_norm,
            x_norm=x_norm,
            x_bias=x_bias,
            condition_on_fg=condition_on_fg,
        )

        self.lr = lr
        self.clip_grad = clip_grad
        if self.clip_grad:
            self.gradient_norm_queue = deque([3000.0], maxlen=50)
        self.validation_metrics = None
        self.molecule_builder = MoleculeBuilder(include_invalid=True)

        self.analyse_samples_every_n_steps = 25000
        self.next_analyse_samples = self.analyse_samples_every_n_steps
        self._run_validation = False

    def setup_metrics(self, train_smiles):
        self.validation_metrics = torch.nn.ModuleDict(
            {
                "Novelty": MolecularNovelty(train_smiles),
                "Validity": MolecularValidity(),
                "Connectivity": MolecularConnectivity(),
                "Lipinski": MolecularLipinski(),
                "LogP": MolecularLogP(),
                "QED": MolecularQEDValue(),
                "SAScore": MolecularSAScore(),
            }
        )

    @skip_computation_on_oom(
        return_value=None, error_message="Skipping batch due to OOM"
    )
    def training_step(self, batch, batch_idx):
        (
            loss,
            loss_unweighted,
            pos_mse,
            x_mse,
        ) = self.consistency_model(batch)
        self.log("loss/train", loss, batch_size=batch.num_graphs)
        self.log("pos_mse/train", pos_mse, batch_size=batch.num_graphs)
        self.log("x_mse/train", x_mse, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._run_validation = self.global_step > self.next_analyse_samples

    def validation_step(self, batch, batch_idx):
        (
            loss,
            loss_unweighted,
            pos_mse,
            x_mse,
        ) = self.consistency_model(batch)
        self.log("loss/val", loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log("pos_mse/val", pos_mse, batch_size=batch.num_graphs, sync_dist=True)
        self.log("x_mse/val", x_mse, batch_size=batch.num_graphs, sync_dist=True)
        if self._run_validation:
            self.analyse_samples(batch, batch_idx)

        return loss

    def analyse_samples(self, batch, batch_idx):

        samples = self.consistency_model.sample(batch)[-1]
        molecules = self.molecule_builder(samples)
        for k, metric in self.validation_metrics.items():
            metric(molecules)
            self.log(
                f"{k}/val",
                metric,
                batch_size=batch.num_graphs,
                sync_dist=True,
            )
        self.log_molecule_visualizations(molecules, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self._run_validation:
            self.next_analyse_samples = (
                self.global_step
                - (self.global_step % self.analyse_samples_every_n_steps)
                + self.analyse_samples_every_n_steps
            )

    def log_molecule_visualizations(self, molecules, batch_idx):
        images = []
        captions = []
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
            img = image_to_tensor(Draw.MolToImage(mol, size=(500, 500)))
            images.append(img)
            captions.append(f"{self.current_epoch}_{batch_idx}_{i}")

        grid_image = make_grid(images)
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    f"log_image_{batch_idx}",
                    grid_image,
                    self.current_epoch,
                )
            if isinstance(logger, pl.loggers.WandbLogger):
                logger.log_image(key="test_set_images", images=images, caption=captions)
        # log_dir = Path(self.logger.log_dir) / "samples"
        # log_dir.mkdir(exist_ok=True, parents=True)
        # Draw.MolToFile(mol, f"{log_dir}/{self.current_epoch}_{batch_idx}_{i}.png")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12
        )
        return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        # optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * (standard deviation of the recent history).
        max_grad_norm: float = 1.5 * np.mean(self.gradient_norm_queue) + 2 * np.std(
            self.gradient_norm_queue
        )
        # Get current grad_norm
        grad_norm = float(get_grad_norm(optimizer))
        self.gradient_norm_queue.append(min(grad_norm, max_grad_norm))
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        """
        results = self.trainer._results
        if isinstance(results.batch, HeteroData):
            results.batch_size = results.batch.num_graphs
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


#TODO added save code
    def save_pretrained(self, save_directory):
        # Make sure the save directory exists
        os.makedirs(save_directory, exist_ok=True)
        # Path to save the model file
        model_file_path = os.path.join(save_directory, "model.pt")
        # Save the model state dictionary
        torch.save(self.state_dict(), model_file_path)

def get_grad_norm(
    optimizer: torch.optim.Optimizer, norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )

    return total_norm
