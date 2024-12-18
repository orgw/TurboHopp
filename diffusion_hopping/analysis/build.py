from pathlib import Path
from tempfile import NamedTemporaryFile

from rdkit import Chem
from torch_geometric.data import HeteroData
from rdkit.Chem import rdmolops

from diffusion_hopping.data import Ligand
from diffusion_hopping.data.featurization.util import atom_names
from diffusion_hopping.data.transform import ObabelTransform


class MoleculeBuilder:
    def __init__(
        self,
        include_invalid=False,
        sanitize=True,
        removeHs=True,
        fix_hydrogens=True,
        atom_names=atom_names,
        refine_molecule = False,

    ):
        self.include_invalid = include_invalid
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.should_fix_hydrogens = fix_hydrogens
        self.xyz_to_sdf = ObabelTransform(from_format="xyz", to_format="sdf")
        self.atom_names = atom_names
        self.refine_molecule = refine_molecule
    def __call__(self, x):
        molecules = []
        if self.refine_molecule:
            for item in x.to_data_list():
                try:
                    mol = self.build_mol(item["ligand"])
                    refined_mol = self.refine_mol(mol)
                    molecules.append(refined_mol)
                except ValueError as e:
                    if self.include_invalid:
                        molecules.append(None)
        else:
            for item in x.to_data_list():
                try:
                    mol=self.build_mol(item["ligand"])
                    molecules.append(mol)
                except ValueError as e:
                    if self.include_invalid:
                        molecules.append(None)
        return molecules

    def build_mol(self, ligand: HeteroData) -> Chem.Mol:
        """Build molecules from HeteroData"""
        xyz_path = self.xyz_from_hetero_data(ligand)
        sdf_path = self.xyz_to_sdf(xyz_path)
        xyz_path.unlink()
        mol = self.mol_from_sdf(sdf_path)
        sdf_path.unlink()
        if self.should_fix_hydrogens:
            self.fix_hydrogens(mol)

        return mol

    def xyz_from_hetero_data(self, x: HeteroData) -> Path:
        """Build xyz from HeteroData"""
        pos = x.pos.detach().cpu().numpy()
        types = x.x.detach().cpu().argmax(axis=-1).numpy()
        types = [self.atom_names[t] for t in types]
        return self.write_xyz_file(pos, types)

    def mol_from_sdf(self, sdf_path: Path) -> Chem.Mol:
        """Build molecule from sdf file"""
        return Ligand(sdf_path).rdkit_mol(
            sanitize=self.sanitize, removeHs=self.removeHs
        )

    @staticmethod
    def write_xyz_file(pos, atom_type) -> Path:
        with NamedTemporaryFile("w", delete=False) as f:
            f.write(f"{len(pos)}\n")
            f.write("generated by model\n")
            for pos, atom_type in zip(pos, atom_type):
                f.write(f"{atom_type} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:,.9f}\n")
            return Path(f.name)

    @staticmethod
    def fix_hydrogens(mol: Chem.Mol):
        organicSubset = (5, 6, 7, 8, 9, 15, 16, 17, 35, 53)
        for at in mol.GetAtoms():
            if at.GetAtomicNum() not in organicSubset:
                continue
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(0)
            at.SetNumRadicalElectrons(0)
        Chem.SanitizeMol(mol)
        return mol
    

        # Function to find the closest atoms between any two fragments that can be connected
    def find_closest_atoms(self, mol, frag_indices):
        conf = mol.GetConformer()
        min_distance = float('inf')
        closest_atoms = (None, None)

        # Iterate over all pairs of fragments
        for i in range(len(frag_indices)):
            for j in range(i+1, len(frag_indices)):
                for idx1 in frag_indices[i]:
                    for idx2 in frag_indices[j]:
                        # Skip if atoms do not have an open valence
                        if not (mol.GetAtomWithIdx(idx1).GetImplicitValence() > 0 and
                                mol.GetAtomWithIdx(idx2).GetImplicitValence() > 0):
                            continue

                        # Calculate distance between the two atoms
                        pos1 = conf.GetAtomPosition(idx1)
                        pos2 = conf.GetAtomPosition(idx2)
                        distance = pos1.Distance(pos2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_atoms = (idx1, idx2)

        return closest_atoms

    def can_add_bond(self, mol, idx1, idx2):
        # Simplified check. Expand as needed based on atom types, current bonds, etc.
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)

        # Check if both atoms have room for an additional bond
        if atom1.GetImplicitValence() > 0 and atom2.GetImplicitValence() > 0:
            return True
        return False

    def refine_mol(self, mol):
        try:
            frag_indices = rdmolops.GetMolFrags(mol, asMols=False)
            if len(frag_indices) == 1:
                return mol  # Already a single fragment

            rw_mol = Chem.RWMol(mol)
            while len(rdmolops.GetMolFrags(rw_mol, asMols=False)) > 1:
                atom1_idx, atom2_idx = self.find_closest_atoms(rw_mol, frag_indices)
                if atom1_idx is None or atom2_idx is None:
                    break  # No suitable pair found

                if self.can_add_bond(rw_mol, atom1_idx, atom2_idx):
                    rw_mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(rw_mol)  # Ensure molecule is still valid
                else:
                    break  # Cannot add more bonds without exceeding valence

            return rw_mol
        except Exception as e:
            print(f"An error occurred: {e}. Returning original molecule.")
            return mol