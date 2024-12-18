import os
import subprocess
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
import subprocess
from meeko import MoleculePreparation, PDBQTWriterLegacy
from queue import Queue
from threading import Thread, Lock

from utils._util_consistency import *
# from posecheck import PoseCheck
import logging
import os 
import pickle
from rdkit import Chem
from tqdm import tqdm


def calculate_rmsd(ref_mol, test_mol):
    """Calculate RMSD between two molecules."""
    # Align the molecules
    ref_mol = Chem.RemoveHs(ref_mol)
    test_mol = Chem.RemoveHs(test_mol)
    AllChem.AlignMol(test_mol, ref_mol)
    
    # Calculate RMSD
    rmsd = AllChem.GetBestRMS(ref_mol, test_mol)
    return rmsd
def _run_commands(commands: List[str], cwd=None) -> str:
    """Executes a list of shell commands."""
    commands_str = " && ".join(commands)
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", commands_str],
            text=True,
            capture_output=True,
            check=True,
            cwd=cwd
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Command execution failed with return code {e.returncode}."
        error_details = f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        raise RuntimeError(f"{error_message}\n{error_details}")

def setup_directory(base_dir, identifier):
    """Create and return directory path based on identifier."""
    path = os.path.join(base_dir, identifier)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def calculate_center_of_mass(mol):
    """Calculate the center of mass for the given molecule."""
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    weights = [atom.GetMass() for atom in mol.GetAtoms()]
    center_of_mass = np.average(coords, axis=0, weights=weights)
    return center_of_mass

def calculate_distance_between_centers(original_mol, modified_mol):
    """Calculate the Euclidean distance between the centers of mass of two molecules."""
    center1 = calculate_center_of_mass(original_mol)
    center2 = calculate_center_of_mass(modified_mol)
    distance = np.linalg.norm(center1 - center2)
    return distance
    
def translate_molecule(mol, translation_vector):
    """Translate molecule's coordinates by the given vector."""
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (x + translation_vector[0], y + translation_vector[1], z + translation_vector[2]))

def minimally_modify_molecule(mol):
    """Modify the molecule minimally by adding hydrogens and optimizing its geometry, recentering it afterwards."""
    original_mol = Chem.Mol(mol)  # Make a copy for comparison
    mol_with_hs = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol_with_hs)
    # Recenter the molecule
    original_center = calculate_center_of_mass(original_mol)
    modified_center = calculate_center_of_mass(mol_with_hs)
    translation_vector = original_center - modified_center
    translate_molecule(mol_with_hs, translation_vector)
    return mol_with_hs, original_mol

def create_sdf_from_mols(molecules,cfg):
    sdf_files = []
    for idx, mol in enumerate(molecules):
        output_path = f"{cfg.gen_mol_dir}/molecule_{idx+1}.sdf"
        if mol is not None:
            mol = Chem.AddHs(mol)
            # etkdgv3 = rdDistGeom.ETKDGv3()
            # if rdDistGeom.EmbedMolecule(mol, etkdgv3) == 0:
            #     try:
            #         rdForceFieldHelpers.UFFOptimizeMolecule(mol)
            #         # pdbqt_mols.append(mol)
            #     except:
            #         print("UFF optimization failed, skipping molecule.")
            #         sdf_files.append('')
            #         continue
            writer = Chem.SDWriter(output_path)
            writer.write(mol)
            writer.close()
            sdf_files.append(output_path)
        else: 
            sdf_files.append('')
    return sdf_files

def prepare_pdbqt_from_sdf(sdf_path, output_path):
    """Prepare a PDBQT file from an SDF file."""
    # if os.path.exists(output_path):
    #     print(f"molecule PDBQT file already exists at {output_path}, skipping.")
    #     return
    mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = mol_supplier[0] if mol_supplier else None
    if mol is None:
        print(f"Error loading or empty SDF file at {sdf_path}")
        return None
    try:
        preparator = MoleculePreparation(rigid_macrocycles=True)
        prepared_mol = preparator.prepare(mol)[0]
        pdbqt_string = PDBQTWriterLegacy.write_string(prepared_mol)
        if pdbqt_string[0] == '' and 'implicit hydrogens' in pdbqt_string[2]:
            mol_with_hs, original_mol = minimally_modify_molecule(mol)
            distance = calculate_distance_between_centers(original_mol, mol_with_hs)
            print(f"Distance between centers after modification: {distance:.4f} Å")
            preparator = MoleculePreparation(rigid_macrocycles=True)
            prepared_mol = preparator.prepare(mol_with_hs)[0]
            pdbqt_string = PDBQTWriterLegacy.write_string(prepared_mol)

        with open(output_path, 'w') as f:
            f.write(pdbqt_string[0])
    except Exception as e:
        print(f"Failed to prepare PDBQT file from SDF: {e}")
        return None

def prepare_protein_pdbqt(pdb_path, output_path):
    """Prepare a PDBQT file from a protein PDB file."""
    if os.path.exists(output_path):
        print(f"Protein PDBQT file already exists at {output_path}, skipping.")
        return
    commands_prot = [
        'eval "$(conda shell.zsh hook)"',
        "conda activate mgltools",
        f"/home/ubuntu/anaconda3/envs/mgltools/bin/python /home/ubuntu/anaconda3/envs/mgltools/bin/prepare_receptor4.py -r {pdb_path} -o {output_path}",
    ]
    try:
        output = _run_commands(commands_prot)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected protein PDBQT file was not created at {output_path}")
    except RuntimeError as e:
        print(f"Error processing {pdb_path}, attempting fix.")
        try:
            commands_fix = [
                "source /home/ubuntu/anaconda3/etc/profile.d/conda.sh",
                "conda activate diffhopp_rl",
                f"pdbfixer {pdb_path} --output {pdb_path[:-4]}_fixed.pdb"
            ]
            _run_commands(commands_fix)
            commands_prot_fixed = [
                'eval "$(conda shell.zsh hook)"', 
                "conda activate mgltools",
                f"/home/ubuntu/anaconda3/envs/mgltools/bin/python /home/ubuntu/anaconda3/envs/mgltools/bin/prepare_receptor4.py -r {pdb_path[:-4]}_fixed.pdb -o {output_path}",
            ]
            _run_commands(commands_prot_fixed)
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Expected fixed protein PDBQT file was not created at {output_path}")
        except RuntimeError as e:
            print(f"Failed to fix and process {pdb_path}: {e}")
            raise
def generate_gpf_and_run_autogrid(rec_path, lig_path, pad, result_dir):
    """Generate grid parameter file (GPF) and run autogrid."""
    if not os.path.exists(result_dir):
        print(f"Result directory does not exist: {result_dir}")
        return  # Exit or handle as necessary

    original_cwd = os.getcwd()
    os.chdir(result_dir)

    gpf_filename = os.path.splitext(os.path.basename(rec_path))[0] + '.gpf'
    gpf_path = os.path.join(result_dir, gpf_filename)
    if os.path.exists(gpf_path):
        print(f"gpf file already exists at {gpf_path}, skipping.")
        return
    if not os.path.exists(rec_path) or not os.path.exists(lig_path):
        print(f"Receptor or ligand PDBQT file not found. Receptor: {rec_path}, Ligand: {lig_path}")
        os.chdir(original_cwd)
        return

    generate_gpf_command = [
        'python', '/home/ubuntu/kiwoong/diffusion-hopping/sejeong/write-gpf.py',
        rec_path, '--lig', lig_path, '--pad', str(pad), '--result_dir', './'
    ]
    try:
        gpf_result = subprocess.run(generate_gpf_command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"GPF generation failed for {rec_path}. STDERR: {e.stderr}")
        os.chdir(original_cwd)
        return

    if not os.path.exists(gpf_path):
        print(f"GPF file not found after generation: {gpf_path}")
        os.chdir(original_cwd)
        return

    autogrid_command = ['autogrid4', '-p', gpf_path, '-l', gpf_path.replace('.gpf', '.glg')]
    autogrid_result = subprocess.run(autogrid_command, capture_output=True, text=True, check=True)

    if not os.path.exists(gpf_path.replace('.gpf', '.glg')):
        print(f"AutoGrid log file not created: {gpf_path.replace('.gpf', '.glg')}")
        os.chdir(original_cwd)
        return

    os.chdir(original_cwd)  # Ensure to switch back to the original directory


def collect_successful_datasets(new_testdataset, cfg):
    successful_datasets = []
    for i in range(len(new_testdataset)):
        identifier = new_testdataset[i]['identifier']
        ref_protein_dir = os.path.join(cfg['ref_pocket_dir'], identifier)
        ref_pocket_maps_path = os.path.join(ref_protein_dir, 'ref_prot.maps.fld')

        if os.path.exists(ref_pocket_maps_path):
            successful_datasets.append(new_testdataset[i])

    return successful_datasets


def dock_molecule_on_gpu(protein_maps, ligand_pdbqt_path, output_path, gpu_id, autodock_path="/home/ubuntu/kiwoong/diffusion-hopping/AutoDock-GPU/bin/autodock_gpu_256wi"):
    """Run docking on a single molecule using a specified GPU and return the rank 1 binding energy."""
    dlg_file = output_path + '.dlg'
    try:
        subprocess.run([
            autodock_path, '--ffile', protein_maps,
            '--lfile', ligand_pdbqt_path, '--resnam', output_path,
            '--devnum', str(gpu_id + 1)  # Device number starts from 1
        ], stdout=subprocess.PIPE, check=True)
        return find_rank1_energy(dlg_file)
    except subprocess.CalledProcessError:
        return 0

def find_rank1_energy(file_path):
    """Parse the output file to find the rank 1 binding energy."""
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith("DOCKED: USER    Estimated Free Energy of Binding"):
                    energy = float(line.split('=')[1].strip().split(' ')[0])
                    return energy
    except Exception:
        return 0

def multi_gpu_docking_ref(new_testdataset, cfg,  mol_dir_path, num_gpus=8,):
    """Function to perform docking using multiple GPUs and return scores in the original order."""
    scores = [None] * len(new_testdataset)  # Preallocate list for scores
    lock = Lock()  # Lock for thread-safe operations on the scores list

    def worker(data_queue, gpu_id):
        while not data_queue.empty():
            index, data = data_queue.get()
            identifier = data['identifier']
            ref_protein_dir = os.path.join(cfg['ref_pocket_dir'], identifier)
            mol_dir = os.path.join(mol_dir_path, identifier)

            ref_pocket_maps_path = os.path.join(ref_protein_dir, 'ref_prot.maps.fld')
            ref_lig_pdbqt_path = os.path.join(mol_dir, 'ref_mol.pdbqt')
            output_path = os.path.join(mol_dir, f'docking_results_gpu{gpu_id}')

            if os.path.exists(ref_pocket_maps_path) and os.path.exists(ref_lig_pdbqt_path):
                energy = dock_molecule_on_gpu(ref_pocket_maps_path, ref_lig_pdbqt_path, output_path, gpu_id)
                with lock:
                    scores[index] = energy
                # tqdm.write(f"Docking completed for {identifier} on GPU {gpu_id}, Rank 1 Energy: {energy} kcal/mol")
            else:
                tqdm.write(f"No valid docking results for {identifier} on GPU {gpu_id}. Score: 0")

    data_queue = Queue()
    for i, data in enumerate(new_testdataset):
        data_queue.put((i, data))

    progress = tqdm(total=len(new_testdataset), desc="Docking reference ligands")
    threads = [Thread(target=worker, args=(data_queue, i)) for i in range(num_gpus)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    progress.close()
    print("All docking tasks completed using multiple GPUs.")
    return scores



# successful_datasets = collect_successful_datasets(new_testdataset, cfg)
# print(f"Number of successful datasets: {len(successful_datasets)}")


# # Configuration and paths
# cfg = {
#     'ref_pocket_dir': "/data/aigen/consistency/ppo/ref/prot",
#     'ref_mol_dir': "/data/aigen/consistency/ppo/ref/mol"
# }

# def multi_gpu_docking_custom(new_testdataset, cfg, mol_dir_path, num_gpus=8):
#     """Function to perform docking using multiple GPUs and return scores in the original order."""
#     scores = [None] * len(new_testdataset)  # Preallocate list for scores
#     lock = Lock()  # Lock for thread-safe operations on the scores list

#     def worker(data_queue, gpu_id):
#         while not data_queue.empty():
#             index, data = data_queue.get()
#             identifier = data['identifier']
#             ref_protein_dir = os.path.join(cfg['ref_pocket_dir'], identifier)
#             mol_dir = os.path.join(mol_dir_path, identifier)

#             ref_pocket_maps_path = os.path.join(ref_protein_dir, 'ref_prot.maps.fld')
#             ref_lig_pdbqt_path = os.path.join(mol_dir, 'ref_mol.pdbqt')
#             output_path = os.path.join(mol_dir, f'docking_results_gpu{gpu_id}')

#             if os.path.exists(ref_pocket_maps_path) and os.path.exists(ref_lig_pdbqt_path):
#                 energy = dock_molecule_on_gpu(ref_pocket_maps_path, ref_lig_pdbqt_path, output_path, gpu_id, cfg['autodock_path'])
#                 with lock:
#                     scores[index] = energy
#             else:
#                 tqdm.write(f"No valid docking results for {identifier} on GPU {gpu_id}. Score: 0")

#     data_queue = Queue()
#     for i, data in enumerate(new_testdataset):
#         data_queue.put((i, data))

#     progress = tqdm(total=len(new_testdataset), desc="Docking custom ligands")
#     threads = [Thread(target=worker, args=(data_queue, i)) for i in range(num_gpus)]
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()

#     progress.close()
#     print("All docking tasks completed using multiple GPUs.")
#     return scores


def multi_gpu_docking_custom(new_testdataset, cfg, mol_dir_path, num_gpus=8):
    """Function to perform docking using multiple GPUs and return scores in the original order."""
    scores = [None] * len(new_testdataset)  # Preallocate list for scores
    lock = Lock()  # Lock for thread-safe operations on the scores list
    progress = tqdm(total=len(new_testdataset), desc="Docking custom ligands")

    def worker(data_queue, gpu_id):
        while not data_queue.empty():
            index, data = data_queue.get()
            identifier = data['identifier']
            ref_protein_dir = os.path.join(cfg['ref_pocket_dir'], identifier)
            mol_dir = os.path.join(mol_dir_path, identifier)

            ref_pocket_maps_path = os.path.join(ref_protein_dir, 'ref_prot.maps.fld')
            ref_lig_pdbqt_path = os.path.join(mol_dir, 'ref_mol.pdbqt')
            output_path = os.path.join(mol_dir, f'docking_results_gpu{gpu_id}')

            if os.path.exists(ref_pocket_maps_path) and os.path.exists(ref_lig_pdbqt_path):
                energy = dock_molecule_on_gpu(ref_pocket_maps_path, ref_lig_pdbqt_path, output_path, gpu_id, cfg['autodock_path'])
                with lock:
                    scores[index] = energy
                with lock:
                    progress.update(1)
            else:
                tqdm.write(f"No valid docking results for {identifier} on GPU {gpu_id}. Score: 0")
                with lock:
                    progress.update(1)

    data_queue = Queue()
    for i, data in enumerate(new_testdataset):
        data_queue.put((i, data))

    threads = [Thread(target=worker, args=(data_queue, i % num_gpus)) for i in range(len(new_testdataset))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    progress.close()
    print("All docking tasks completed using multiple GPUs.")
    return scores

##############posecheck##############

# interaction_scores = {
#     'Hydrophobic': 2.5, 'HBDonor': 3.5, 'HBAcceptor': 3.5, 'Anionic': 7.5, 
#     'Cationic': 7.5, 'CationPi': 2.5, 'PiCation': 2.5, 'VdWContact': 1.0, 
#     'XBAcceptor': 3.0, 'XBDonor': 3.0, 'FaceToFace': 3.0, 'EdgeToFace': 1.0, 
#     'MetalDonor': 3.0, 'MetalAcceptor': 3.0,
# }

# def get_interaction_score(interactions, interaction_scores=interaction_scores):
#     df_stacked = interactions.stack(level=[0, 1, 2], future_stack=True)
#     df_reset = df_stacked.to_frame().reset_index()
#     df_reset.columns = ['Frame', 'ligand', 'protein', 'interaction', 'value']
#     df_reset['score'] = df_reset['interaction'].apply(lambda x: interaction_scores.get(x, 0))
#     return df_reset['score'].sum()


def process_data(entry, ligand, posecheck_objects):
    identifier = entry['identifier']
    pc = posecheck_objects[identifier]  # Use pre-loaded PoseCheck object
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