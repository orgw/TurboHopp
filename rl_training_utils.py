from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import tqdm
from openbabel import openbabel
from rdkit import RDLogger
from tqdm import tqdm
from models_consistency import *
from train_consistency import *
from _util_consistency import *
from torch_scatter import scatter_mean
import itertools
import pdb
from multiprocessing import Pool
#############

import numpy as np
from collections import deque
###########docking tools#############
from docking_posecheck_utils import *
import multiprocessing

class PerIDStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, ids, rewards):
        ids = ids.cpu()
        rewards = rewards.cpu()
        ids = np.array(ids)
        rewards = np.array(rewards)
        unique_ids = np.unique(ids, axis=0)
        advantages = np.zeros_like(rewards)

        for identifier in unique_ids:
            tf_list = [(id == identifier).all() for id in ids]
            id_rewards = rewards[tf_list]
            if str(identifier.tolist()) not in self.stats:
                self.stats[str(identifier.tolist())] = deque(maxlen=self.buffer_size)
            self.stats[str(identifier.tolist())].extend(id_rewards)

            if len(self.stats[str(identifier.tolist())]) < self.min_count:
                # Calculate mean and std using all rewards if not enough samples for this identifier
                mean = np.mean(rewards) if len(rewards) > 0 else 0
                std = np.std(rewards) if len(rewards) > 0 else 1e-6
            else:
                mean = np.mean(self.stats[str(identifier.tolist())])
                std = np.std(self.stats[str(identifier.tolist())]) + 1e-6

            advantages[tf_list] = (id_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            id: {"mean": np.mean(stats), "std": np.std(stats), "count": len(stats)}
            for id, stats in self.stats.items()
        }


# # Example usage
# tracker = PerIDStatTracker(buffer_size=32, min_count=16)
# ids = ['1abc', '2def', '1abc', '3ghi', '2def', '1abc']
# rewards = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# advantages = tracker.update(ids, rewards)
# print("Advantages:", advantages)
# print("Statistics:", tracker.get_stats())

############################


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    # log_prob = (
    #     -((values - means) ** 2) / (2 * var.unsqueeze(-1))
    #     - log_scales.unsqueeze(-1)
    #     - np.log(np.sqrt(2 * np.pi))
    # )

    norm_factor = np.log(np.sqrt(2 * np.pi))
    term1 = -((values - means) ** 2) / (2 * var.unsqueeze(-1))
    term2 = -log_scales.unsqueeze(-1)
    term3 = -norm_factor
    log_prob = term1 + term2 + term3
    # print("Term1:", term1)
    # print("Term2:", term2)
    # print("Term3:", term3)
    # print("Log probability components:", log_prob)
    if torch.any(log_prob.mean(dim=tuple(range(1, log_prob.ndim))) ==0):
        import pdb; pdb.set_trace()
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

def log_normal_reduce(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = (
        -((values - means) ** 2) / (2 * var.unsqueeze(-1))
        - log_scales.unsqueeze(-1)
        - np.log(np.sqrt(2 * np.pi))
    )
    return log_prob.mean()


############################
def sample_from_cm(model, x_0, sampling_sigmas, mask, device, cfg):
    model.eval()
    
    with torch.no_grad():
        state_list = []
        log_prob_list = []

        # TODO Initialize x and pos with random noise.
        x = x_0.clone()
        

        x["protein"].x = x["protein"].x.to(dtype=torch.float)
        x["protein"].pos = x["protein"].pos.to(dtype=torch.float)
        x["ligand"].x[mask] = torch.randn_like(x_0["ligand"].x[mask], device=device)
        x["ligand"].pos[mask] = util.centered_batch(
            torch.randn_like(x["ligand"].pos[mask], device=device),
            x["ligand"].batch[mask],
            dim_size=x.num_graphs,
        )  

        x = x.to(device)

        state_list.append(x.clone())

    
        # TODO accelerator?? this for loop does :  get sigma(s.d) -> add noise -> denoise(get mean) -> get log probs and add ..
        for i, sigma in tqdm(
            enumerate(sampling_sigmas),
            desc="Processing",
            total=len(sampling_sigmas),
        ):  
            
            # create list of shape (x["ligand"].x[mask].shape[0],)
            sigma_full = torch.full((x["ligand"].x[mask.cpu()].shape[0],), sigma, dtype=torch.float, device=device)
            # # SAMPLE
            # print("model device:", model.device)
            # # print("x device:", x.device)
            # print("mask device:", mask.device)
            #################################################testing

            # a0 = x.clone()
            # a2 = x.clone()
            # model.consistency_model = model.consistency_model.eval()
            # model.consistency_model.estimator = model.consistency_model.estimator.eval()

            # a1 = x.clone()
            # a3 = x.clone()

            
            # a0["ligand"].x[mask], a0["ligand"].pos[mask] = model_forward_wrapper_difsigma(model, a0, mask, sigma_full.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min)
            # import pdb; pdb.set_trace()
            
            # a2["ligand"].x[mask], a2["ligand"].pos[mask] = model_forward_wrapper_difsigma(model, a2, mask, sigma_full.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min)
            # import pdb; pdb.set_trace()

            # a1["ligand"].x[mask], a1["ligand"].pos[mask] = model_forward_wrapper(
            #     model, a1, mask, sigma_full.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min
            # )
            # a3["ligand"].x[mask], a3["ligand"].pos[mask] = model_forward_wrapper(
            #     model, a3, mask, sigma_full.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min
            # )
            # import pdb; pdb.set_trace()

            x["ligand"].x[mask], x["ligand"].pos[mask] = model_forward_wrapper_difsigma(
                model, x, mask, sigma_full.unsqueeze(-1), cfg.sigma_data, cfg.sigma_min
            )

            x["ligand"].pos[mask] = util.centered_batch(
                x["ligand"].pos[mask], x["ligand"].batch[mask], dim_size=x.num_graphs
            )
            no_noise_x = x.clone()
            # ADD NOISE

            x_eps = torch.randn_like(x["ligand"].x[mask.cpu()], device=device)
            pos_eps = util.centered_batch(
                torch.randn_like(x["ligand"].pos[mask.cpu()], device=device),
                x["ligand"].batch[mask.cpu()],
                dim_size=x.num_graphs,
            )

            if i == len(sampling_sigmas) - 1:
                sigma_diff = torch.zeros_like(sigma_full)
            else:
                sigma_diff = ((sigma_full**2 - cfg.sigma_min**2) ** 0.5)
            x["ligand"].x[mask.cpu()] = (
                x["ligand"].x[mask.cpu()].to(dtype=torch.float)
                + pad_dims_like(
                    sigma_diff.unsqueeze(-1), x["ligand"].x[mask.cpu()].to(dtype=torch.float)
                )
                * x_eps
            )
            x["ligand"].pos[mask.cpu()] = (
                x["ligand"].pos[mask.cpu()].to(dtype=torch.float)
                + pad_dims_like(
                    sigma_diff.unsqueeze(-1), x["ligand"].pos[mask.cpu()].to(dtype=torch.float)
                )
                * pos_eps
            )

            log_prob_x = log_normal(
                no_noise_x["ligand"].x[mask.cpu()],
                means=x["ligand"].x[mask.cpu()],
                log_scales=torch.log(sigma_full),
            )
            log_prob_pos = log_normal(
                no_noise_x["ligand"].pos[mask.cpu()],
                means=x["ligand"].pos[mask.cpu()],
                log_scales=torch.log(sigma_full),)
            # )
            # print(scatter_mean(log_prob_x, x["ligand"].batch[mask], dim=0))
            # print(scatter_mean(log_prob_pos, x["ligand"].batch[mask], dim=0))
            # print('.')
            # if torch.any((scatter_mean(log_prob_x, x["ligand"].batch[mask], dim=0) + scatter_mean(log_prob_pos, x["ligand"].batch[mask], dim=0)) == 0):
            #     import pdb; pdb.set_trace()

            log_prob_list.append(
                scatter_mean(log_prob_x, x["ligand"].batch[mask], dim=0)
                + scatter_mean(log_prob_pos, x["ligand"].batch[mask], dim=0)
            )
            state_list.append(x.clone())

        lp_list = torch.stack(log_prob_list, dim=-1)

        # leave out last column


        state_list = [[j[i] for j in state_list] for i in range(len(state_list[0]))]
        state_list_but_last = [state_list[i][:-1] for i in range(len(state_list))]
        state_list_after_last = [state_list[i][1:] for i in range(len(state_list))]

        return {
            "id": x_0["identifier"],
            "log_probs": lp_list,
            "states": state_list_but_last,  # each entry is the molecule before timestep t
            "next_states": state_list_after_last,  # each entry is the molecule after timestep t
            "generated_mols": model.molecule_builder(no_noise_x),
            "sigmas": torch.tensor(sampling_sigmas[:]).repeat(len(state_list), 1).to(device),
        }

#############################

def calculate_unified_score_docking(qed_score, sa_raw_score, docking_score):
    """Normalize SA score and include docking score in the unified score."""
    if qed_score is None or sa_raw_score is None:
        return None  # Return a default score, such as 0
    sa_score_normalized = (10 - sa_raw_score) / 9
    return (qed_score + sa_score_normalized - docking_score) / 3  # Averaging and considering docking score as negative
def calculate_unified_score(qed_score, sa_raw_score):
    """Normalize SA score and calculate the unified score as an average of QED and normalized SA."""
    if qed_score is None or sa_raw_score is None:
        print("score is None")
        return None  # or return some default score, e.g., 0
    sa_score_normalized = (10 - sa_raw_score) / 9
    return (qed_score + sa_score_normalized) / 2

def connectivity_score(mol):
    """Return the number of fragments in the molecule."""
    if len(Chem.GetMolFrags(mol, asMols=True)) == 1:
        return 1  # Reward of 1 for a single, whole molecule
    else:
        return 0  # No reward for fragmented molecules

def mols_to_pdbqts(molecules, identifiers, cfg):
    valid_identifiers = []
  
    if not os.path.exists(os.path.join(cfg['gen_mol_dir'])):
        os.makedirs(os.path.join(cfg['gen_mol_dir']))  
    print("saving sdfs at :::", os.path.join(cfg['gen_mol_dir']))
    sdf_file_paths = create_sdf_from_mols(molecules, cfg)
    failed_pdbqt_attempts =0
    empty_mols = 0
    assert len(molecules) == len(sdf_file_paths)

    for idx, (sdf_file, identifier) in enumerate(zip(sdf_file_paths, identifiers)):
        if molecules[idx]:
            try:
                pdbqt_dir = os.path.join(cfg['gen_mol_dir'], identifier)
                if not os.path.exists(pdbqt_dir):
                    os.makedirs(pdbqt_dir)
                pdbqt_path = os.path.join(pdbqt_dir, "ref_mol.pdbqt")
                prepare_pdbqt_from_sdf(sdf_file, pdbqt_path)
                valid_identifiers.append(identifier)
            except:
                failed_pdbqt_attempts+=1
                continue
        else:
            empty_mols+=1
    print("failed converting pdbqts: ", failed_pdbqt_attempts)
    print("empty mols: ", empty_mols)

    return valid_identifiers

def reward_function_docking(mols, identifiers, ref_scores, cfg):
    """Calculate rewards for a list of molecules based on their docking, QED, and SA scores."""
    valid_identifiers = mols_to_pdbqts(mols, identifiers, cfg)
    new_testdataset = [{'identifier': identifier} for identifier in valid_identifiers]
    docking_scores = multi_gpu_docking_custom(new_testdataset, cfg, cfg['gen_mol_dir'], num_gpus=8)

    unified_scores = [0] * len(mols)
    connectivity_scores = [0] * len(mols)
    qed_scores = [0] * len(mols)
    sa_scores = [0] * len(mols)
    final_docking_scores = [0] * len(mols)

    docking_score_map = {identifier: score for identifier, score in zip(valid_identifiers, docking_scores)}

    for i, mol in enumerate(mols):
        if mol is not None and mol.GetNumAtoms() > 0:
            identifier = identifiers[i]
            if identifier in valid_identifiers:
                qed_value = QED.qed(mol)
                sa_raw_score = sascorer.calculateScore(mol)
                connectivity = connectivity_score(mol)
                docking_score = docking_score_map.get(identifier, 0)
                ref_score = ref_scores.get(identifier, 0)

                if ref_score == 0:  # Prevent division by zero
                    final_docking_score = 0
                elif docking_score is None:
                    final_docking_score = 0
                    docking_score = 0
                else:
                    final_docking_score = ((docking_score - ref_score) * 2)

                unified_score = calculate_unified_score(qed_value, sa_raw_score) + connectivity - final_docking_score
                unified_scores[i] = unified_score
                connectivity_scores[i] = connectivity
                qed_scores[i] = qed_value
                sa_scores[i] = sa_raw_score
                final_docking_scores[i] = docking_score
            else:
                # Default values for identifiers not found in valid_identifiers
                unified_scores[i] = 0
                connectivity_scores[i] = 0
                qed_scores[i] = 0
                sa_scores[i] = 0
                final_docking_scores[i] = 0

    return unified_scores, connectivity_scores, qed_scores, sa_scores, final_docking_scores
def process_wrapper(args):
    entry, ligand, posecheck_objects = args
    identifier = entry['identifier']
    pc = posecheck_objects[identifier]  # Assuming posecheck_objects pre-loaded with necessary data
    try:
        pc.load_ligands_from_mols([ligand])  # Pass the ligand as a list with one element
        clashes = pc.calculate_clashes()
        strain = pc.calculate_strain_energy()
        return identifier, {
            "clashes": clashes[0],
            "strain_energy": strain[0]
        }
    except Exception as e:
        logging.error(f"Failed to process {identifier}: {e}")
        return identifier, {"error": str(e)}

def reward_function_posecheck(mols, identifiers, posecheck_objects, ref_values, cfg):
    # Prepare the dataset for processing
    new_testdataset = [{'identifier': identifier} for identifier in identifiers]

    # Create a list of arguments for each molecule to be processed
    args_list = [(entry, mol, posecheck_objects) for entry, mol in zip(new_testdataset, mols)]

    # Optimize pool usage by defining the pool outside the function if this function is called multiple times
    with Pool(60) as pool:
        clash_strain_results = list(tqdm(pool.imap(process_wrapper, args_list), total=len(args_list)))

    # Convert results to a dictionary for quick access
    clash_strain_dict = dict(clash_strain_results)

    # Lists to store scores
    unified_scores = [0] * len(mols)
    connectivity_scores = [0] * len(mols)
    qed_scores = [0] * len(mols)
    sa_scores = [0] * len(mols)

    # Process each molecule
    for i, mol in enumerate(mols):
        identifier = identifiers[i]
        if mol is not None and mol.GetNumAtoms() > 0:
            qed_value = QED.qed(mol)
            sa_raw_score = sascorer.calculateScore(mol)
            connectivity = 1 if len(Chem.GetMolFrags(mol, asMols=True)) == 1 else 0

            result = clash_strain_dict.get(identifier, {})
            clashes = result.get("clashes", 0)
            strain_energy = result.get("strain_energy", 0)

            ref_clashes = ref_values.get(identifier, {}).get("clashes", 0)
            ref_strain_energy = ref_values.get(identifier, {}).get("strain_energy", 0)

            clash_reward = clashes - ref_clashes
            strain_reward = strain_energy - ref_strain_energy

            unified_score = qed_value + (10 - sa_raw_score) / 9 + clash_reward + strain_reward
            unified_scores[i] = unified_score
            connectivity_scores[i] = connectivity
            qed_scores[i] = qed_value
            sa_scores[i] = sa_raw_score
        else:
            unified_scores[i] = 0
            connectivity_scores[i] = 0
            qed_scores[i] = 0
            sa_scores[i] = 0

    return unified_scores, connectivity_scores, qed_scores, sa_scores



# def reward_function_docking(mols, prot_pdbqt_paths, cfg):
#     """Calculate rewards for a list of molecules based on their docking, QED, and SA scores."""
#     pdbqt_paths = mols_to_pdbqts(mols, cfg)
    
#     unified_scores = []
#     connectivity_scores = []
#     qed_scores = []
#     sa_scores = []
#     docking_scores = []

#     # Prepare tuples for docking
#     tasks = [(prot_path, lig_path, mol) for prot_path, lig_path, mol in zip(prot_pdbqt_paths, pdbqt_paths, pdbqt_mols) if mol is not None]

#     # Parallel docking computation
#     with Pool(processes=2) as pool:  # Adjust the number of processes based on your CPU
#         docking_results = list(tqdm(pool.imap(calculate_qvina_score_wrapper, tasks), total=len(tasks), desc="Docking Progress"))

#     # Preprocess docking results to handle NaNs and positive scores
#     docking_results = [0 if np.isnan(score) or score > 0 else score for score in docking_results]
#     print(docking_results)
#     # Map results back to molecules
#     docking_index = 0
#     for i, mol in enumerate(pdbqt_mols):
#         if mol is not None and mol.GetNumAtoms() > 0:
#             qed_value = QED.qed(mol)
#             sa_raw_score = sascorer.calculateScore(mol)
#             connectivity = connectivity_score(mol)

#             if pdbqt_paths[i]:  # There is a valid pdbqt path for docking
#                 docking_score = docking_results[docking_index]
#                 docking_index += 1
#             else:
#                 docking_score = 0  # Default score if docking failed

#             docking_scores.append(docking_score)
#             unified_score = calculate_unified_score(qed_value, sa_raw_score) + connectivity - docking_score

#             unified_scores.append(unified_score)
#             connectivity_scores.append(connectivity)
#             qed_scores.append(qed_value)
#             sa_scores.append(sa_raw_score)
#         else:
#             # Default values for invalid molecules
#             unified_scores.append(0)
#             connectivity_scores.append(0)
#             qed_scores.append(0)
#             sa_scores.append(0)
#             docking_scores.append(0)

#     return unified_scores, connectivity_scores, qed_scores, sa_scores, docking_scores

def connectivity_score(mol):
    """Evaluate connectivity of the molecule."""
    if len(Chem.GetMolFrags(mol, asMols=True)) == 1:
        return 1  # Reward for a single, connected molecule
    else:
        return 0  # Penalty for fragmented molecules
            



def reward_function(mols):
    """Calculate rewards for a list of molecules based on their QED and SA scores."""

    unified_scores = []
    connectivity_scores = []
    qed_scores= []
    sa_scores = []
    for mol in mols:
        if mol is not None:
            qed_value = QED.qed(mol)
            sa_raw_score = sascorer.calculateScore(mol)
            connectivity = connectivity_score(mol) 
            unified_score = calculate_unified_score(qed_value, sa_raw_score) + connectivity

            if unified_score is not None:
                unified_scores.append(unified_score)
                connectivity_scores.append(connectivity)
                qed_scores.append(qed_value)
                sa_scores.append(sa_raw_score)

            else:
                unified_scores.append(
                    0
                )  # Assign a default score for molecules with missing data
                connectivity_scores.append(0)
                qed_scores.append(0)
                sa_scores.append(0)

        else:
                unified_scores.append(
                    0
                )  # Assign a default score for molecules with missing data
                connectivity_scores.append(0)
                qed_scores.append(0)
                sa_scores.append(0)

    return unified_scores, connectivity_scores, qed_scores, sa_scores

# def reward_function_new(mols):
#     rewards = []
#     for mol in mols:
#         if mol is not None:
#             qed_value = QED.qed(mol)
#             sa_raw_score = sascorer.calculateScore(mol)
#             connectivity = 1 if len(Chem.GetMolFrags(mol, asMols=True)) == 1 else 0
#             sa_score_normalized = (10 - sa_raw_score) / 9
#             unified_score = (qed_value + sa_score_normalized) / 2 + connectivity 
#             rewards.append({
#                 "qed": qed_value,
#                 "sa_score": sa_raw_score,
#                 "connectivity": connectivity,
#                 "unified_score": unified_score
#             })
#         else:
#             rewards.append({"qed": 0, "sa_score": 0, "connectivity": 0, "unified_score": 0})  # default scores for invalid mols
#     return rewards
#################################

def ids_to_torch(ids):
    # allocate torch tesnor of shape (len(ids),10)
    max_len = max(len(id) for id in ids)
    ids_torch = torch.zeros(len(ids), max_len) #need to increase for crossdocked..
    for i, id in enumerate(ids):
        for j, char in enumerate(id):
            ids_torch[i, j] = ord(char)

    return ids_torch


########### RESHAPE CODE

from functools import reduce 
from operator import mul

def reshape(lst, shape):
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i*n:(i+1)*n], shape[1:]) for i in range(len(lst)//n)]