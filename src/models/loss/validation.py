"""Validation metrics and structure matching utilities."""

from typing import List, Optional, Tuple, Dict, Any
import io
import contextlib

import numpy as np
from pymatgen.core import Structure, Lattice  # Used for find_best_match_rmsd
from pymatgen.analysis.structure_matcher import StructureMatcher
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import Atoms
import sys
from ase.optimize import LBFGS
import warnings

from scripts.assemble import assemble
import smact
from smact.screening import pauling_test
import itertools

# Suppress all warnings for JSON parsing in subprocesses
warnings.filterwarnings('ignore')

# Global calculator for reuse across processes
_GLOBAL_CALC = None

OC20_GAS_PHASE_ENERGIES = {
    'H': -3.477,
    'O': -7.204,
    'C': -7.282,
    'N': -8.083,
}

from pymatgen.io.ase import AseAtomsAdaptor
adaptor = AseAtomsAdaptor()

def find_best_match_rmsd_prim(
    args: Tuple[
        np.ndarray,  # sampled_coords: (n_samples, n_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # true_coords: (n_atoms, 3)
        np.ndarray,  # true_lattices: (6,) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # atom_types: (n_atoms,)
        np.ndarray,  # atom_mask: (n_atoms,) bool
        Dict[str, Any],  # matcher_kwargs
    ],
) -> List[Optional[float]]:
    """
    Find the best RMSD match between sampled structures and the true structure.

    This function is designed to be run in parallel with ProcessPool.

    Args:
        args: Tuple containing:
            - sampled_coords: Sampled atom coordinates (n_samples, n_atoms, 3)
            - sampled_lattices: Sampled lattice parameters (n_samples, 6) - (a, b, c, alpha, beta, gamma)
            - true_coords: True atom coordinates (n_atoms, 3)
            - true_lattices: True lattice parameters (6,) - (a, b, c, alpha, beta, gamma)
            - atom_types: Atomic numbers for each atom (n_atoms,)
            - atom_mask: Boolean mask for valid atoms (n_atoms,)
            - matcher_kwargs: Keyword arguments for StructureMatcher

    Returns:
        List of RMSD values for each sample. None if no match found.
    """
    (
        sampled_coords,
        sampled_lattices,
        true_coords,
        true_lattices,
        atom_types,
        atom_mask,
        matcher_kwargs,
    ) = args

    n_samples = sampled_coords.shape[0]

    # Filter valid atoms
    valid_atom_types = atom_types[atom_mask]
    true_valid_coords = true_coords[atom_mask]

    # Create true structure from lattice parameters (a, b, c, alpha, beta, gamma)
    try:
        true_lattice = Lattice.from_parameters(
            a=true_lattices[0],
            b=true_lattices[1],
            c=true_lattices[2],
            alpha=true_lattices[3],
            beta=true_lattices[4],
            gamma=true_lattices[5],
        )
        true_structure = Structure(
            lattice=true_lattice,
            species=valid_atom_types.tolist(),
            coords=true_valid_coords,
            coords_are_cartesian=True,
        )
    except Exception:
        return [None] * n_samples

    # Create matcher
    matcher = StructureMatcher.from_dict(matcher_kwargs)

    results = []
    for i in range(n_samples):
        try:
            # Create sampled structure from lattice parameters (a, b, c, alpha, beta, gamma)
            sampled_lattice = Lattice.from_parameters(
                a=sampled_lattices[i, 0],
                b=sampled_lattices[i, 1],
                c=sampled_lattices[i, 2],
                alpha=sampled_lattices[i, 3],
                beta=sampled_lattices[i, 4],
                gamma=sampled_lattices[i, 5],
            )
            sampled_valid_coords = sampled_coords[i][atom_mask]

            sampled_structure = Structure(
                lattice=sampled_lattice,
                species=valid_atom_types.tolist(),
                coords=sampled_valid_coords,
                coords_are_cartesian=True,
            )

            # Check if structures match
            if matcher.fit(true_structure, sampled_structure):
                # Compute RMSD if match found
                rmsd = matcher.get_rms_dist(true_structure, sampled_structure)
                if rmsd is not None:
                    results.append(rmsd[0])  # get_rms_dist returns (rms_dist, max_dist)
                else:
                    results.append(None)
            else:
                results.append(None)

        except Exception:
            results.append(None)

    return results


def find_best_match_rmsd_slab(
    args: Tuple[
        np.ndarray,  # sampled_prim_slab_coords: (n_samples, n_prim_slab_atoms, 3)
        np.ndarray,  # sampled_ads_coords: (n_samples, n_ads_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6)
        np.ndarray,  # sampled_supercell_matrices: (n_samples, 3, 3) or (n_samples, 9)
        np.ndarray,  # sampled_scaling_factors: (n_samples,)
        np.ndarray,  # true_prim_slab_coords: (n_prim_slab_atoms, 3)
        np.ndarray,  # true_ads_coords: (n_ads_atoms, 3)
        np.ndarray,  # true_lattices: (6,)
        np.ndarray,  # true_supercell_matrix: (3, 3) or (9,)
        float,       # true_scaling_factor
        np.ndarray,  # prim_slab_atom_types: (n_prim_slab_atoms,)
        np.ndarray,  # ads_atom_types: (n_ads_atoms,)
        np.ndarray,  # prim_slab_atom_mask: (n_prim_slab_atoms,) bool
        np.ndarray,  # ads_atom_mask: (n_ads_atoms,) bool
        Dict[str, Any],  # matcher_kwargs
    ],
) -> List[Optional[float]]:
    """
    Find the best RMSD match between sampled full systems and the true system.
    
    This function assembles the full system (supercell slab + adsorbate) and computes RMSD.
    Designed to be run in parallel with ProcessPool.
    
    Args:
        args: Tuple containing all necessary data for system assembly and RMSD computation.
    
    Returns:
        List of RMSD values for each sample. None if no match found or assembly failed.
    """
    (
        sampled_prim_slab_coords,
        sampled_ads_coords,
        sampled_lattices,
        sampled_supercell_matrices,
        sampled_scaling_factors,
        true_prim_slab_coords,
        true_ads_coords,
        true_lattices,
        true_supercell_matrix,
        true_scaling_factor,
        prim_slab_atom_types,
        ads_atom_types,
        prim_slab_atom_mask,
        ads_atom_mask,
        matcher_kwargs,
    ) = args
    
    n_samples = sampled_prim_slab_coords.shape[0]
    
    # Assemble true system first
    try:
        true_system, true_slab = assemble(
            generated_prim_slab_coords=true_prim_slab_coords,
            generated_ads_coords=true_ads_coords,
            generated_lattice=true_lattices,
            generated_supercell_matrix=true_supercell_matrix.reshape(3, 3),
            generated_scaling_factor=float(true_scaling_factor),
            prim_slab_atom_types=prim_slab_atom_types,
            ads_atom_types=ads_atom_types,
            prim_slab_atom_mask=prim_slab_atom_mask,
            ads_atom_mask=ads_atom_mask,
        )
        
        true_structure = adaptor.get_structure(true_slab)
        
    except Exception:
        return [None] * n_samples
    
    # Create matcher
    matcher = StructureMatcher.from_dict(matcher_kwargs)
    
    results = []
    for i in range(n_samples):
        try:
            # Assemble sampled system
            sampled_system, sampled_slab = assemble(
                generated_prim_slab_coords=sampled_prim_slab_coords[i],
                generated_ads_coords=sampled_ads_coords[i],
                generated_lattice=sampled_lattices[i],
                generated_supercell_matrix=sampled_supercell_matrices[i].reshape(3, 3),
                generated_scaling_factor=float(sampled_scaling_factors[i]),
                prim_slab_atom_types=prim_slab_atom_types,
                ads_atom_types=ads_atom_types,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
            )
            
            sampled_structure = adaptor.get_structure(sampled_slab)
            
            # Check if slabs match
            if matcher.fit(true_structure, sampled_structure):
                # Compute RMSD if match found
                rmsd = matcher.get_rms_dist(true_structure, sampled_structure)
                if rmsd is not None:
                    results.append(rmsd[0])  # get_rms_dist returns (rms_dist, max_dist)
                else:
                    results.append(None)
            else:
                results.append(None)
                
        except Exception:
            results.append(None)
    
    return results

def _structural_validity(atoms: Atoms) -> bool:
    """Check structural validity of an Atoms object."""
    # 1. Check cell volume
    try:
        vol = float(atoms.get_volume())
        vol_ok = vol >= 0.1
    except Exception:
        vol_ok = False

    # 2. Check atom clash
    try:
        if len(atoms) > 1:
            dists = atoms.get_all_distances()
            # Exclude self-distance (0)
            min_dist = np.min(dists[np.nonzero(dists)])
            dist_ok = min_dist >= 0.5
  
        else:
            dist_ok = True  # Skip distance check for single atom
    except Exception:
        dist_ok = False
    
    # 3. Check min width of cell (a, b axes)
    min_ab = 8.0
    
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    
    if a_length < min_ab or b_length < min_ab:
        width_ok = False
    else:
        width_ok = True
    
    # 4. Check min height of cell (c projected onto normal of a×b plane)
    min_height = 20.0
    
    a_vec = atoms.cell[0]
    b_vec = atoms.cell[1]
    c_vec = atoms.cell[2]
    
    # normal = unit vector of (a × b)
    cross_ab = np.cross(a_vec, b_vec)
    cross_ab_norm = np.linalg.norm(cross_ab)
    
    if cross_ab_norm < 1e-10:
        # Degenerate case: a and b are parallel
        print("Height check failed.")
        height_ok = False
    else:
        normal = cross_ab / cross_ab_norm
        proj_height = abs(np.dot(normal, c_vec))
        height_ok = proj_height >= min_height

    # 5. Basic validity check
    basic_valid = bool(vol_ok and dist_ok and width_ok and height_ok)

    return basic_valid

def _prim_structural_validity(atoms: Atoms) -> bool:
    """Check structural validity of an Atoms object."""
    # 1. Check cell volume
    try:
        vol = float(atoms.get_volume())
        vol_ok = vol >= 0.1
    except Exception:
        vol_ok = False

    # 2. Check atom clash
    try:
        if len(atoms) > 1:
            dists = atoms.get_all_distances()
            # Exclude self-distance (0)
            min_dist = np.min(dists[np.nonzero(dists)])
            dist_ok = min_dist >= 0.5
  
        else:
            dist_ok = True  # Skip distance check for single atom
    except Exception:
        dist_ok = False

    # 5. Basic validity check
    basic_valid = bool(vol_ok and dist_ok)

    return basic_valid

def compute_structural_validity_single(
    args: Tuple[
        np.ndarray,  # sampled_coords: (n_samples, n_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # true_coords: (n_atoms, 3)
        np.ndarray,  # true_lattices: (6,) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # atom_types: (n_atoms,)
        np.ndarray,  # atom_mask: (n_atoms,) bool
        Dict[str, Any],  # matcher_kwargs
    ],
) -> List[bool]:
    """
    Compute structural validity for sampled structures (without adsorption energy calculation).
    
    This function reconstructs the full system from prim_slab using supercell_matrix
    and scaling_factor, then checks structural validity.
    
    Designed to be run in parallel with ProcessPool.
    
    Args:
        args: Tuple containing:
            - sampled_prim_slab_coords: Sampled prim slab coordinates (n_samples, n_atoms, 3)
            - sampled_ads_coords: Sampled adsorbate coordinates (n_samples, n_ads_atoms, 3)
            - sampled_lattices: Sampled lattice parameters (n_samples, 6)
            - sampled_supercell_matrices: Supercell transformation matrices (n_samples, 3, 3) or (n_samples, 9)
            - sampled_scaling_factors: Z-direction scaling factors (n_samples,)
            - prim_slab_atom_types: Atomic numbers for prim slab atoms
            - ads_atom_types: Atomic numbers for adsorbate atoms
            - prim_slab_atom_mask: Boolean mask for valid prim slab atoms
            - ads_atom_mask: Boolean mask for valid adsorbate atoms
    
    Returns:
        List of structural validity booleans for each sample.
    """
    (
        sampled_prim_slab_coords,
        sampled_ads_coords,
        sampled_lattices,
        sampled_supercell_matrices,
        sampled_scaling_factors,
        prim_slab_atom_types,
        ads_atom_types,
        prim_slab_atom_mask,
        ads_atom_mask,
    ) = args
    
    n_samples = sampled_prim_slab_coords.shape[0]
    
    results = []
    for i in range(n_samples):
        try:
            # Use assemble function to reconstruct the system
            recon_system, recon_slab = assemble(
                generated_prim_slab_coords=sampled_prim_slab_coords[i],
                generated_ads_coords=sampled_ads_coords[i],
                generated_lattice=sampled_lattices[i],
                generated_supercell_matrix=sampled_supercell_matrices[i].reshape(3, 3),
                generated_scaling_factor=sampled_scaling_factors[i],
                prim_slab_atom_types=prim_slab_atom_types,
                ads_atom_types=ads_atom_types,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
            )
            
            # Check structural validity
            struct_valid = _structural_validity(recon_system)
            results.append(struct_valid)
                
        except Exception as e:
            print(f"WARNING: Failed to compute structural validity for sample {i}: {e}", file=sys.stderr)
            results.append(False)
    
    return results

def compute_prim_structural_validity_single(
    args: Tuple[
        np.ndarray,  # sampled_prim_slab_coords: (n_samples, n_prim_slab_atoms, 3)
        np.ndarray,  # sampled_ads_coords: (n_samples, n_ads_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6)
        np.ndarray,  # sampled_supercell_matrices: (n_samples, 3, 3) or (n_samples, 9)
        np.ndarray,  # sampled_scaling_factors: (n_samples,)
        np.ndarray,  # prim_slab_atom_types: (n_prim_slab_atoms,)
        np.ndarray,  # ads_atom_types: (n_ads_atoms,)
        np.ndarray,  # prim_slab_atom_mask: (n_prim_slab_atoms,) bool
        np.ndarray,  # ads_atom_mask: (n_ads_atoms,) bool
    ],
) -> List[bool]:
    """
    Compute structural validity for sampled structures (without adsorption energy calculation).
    
    This function reconstructs the full system from prim_slab using supercell_matrix
    and scaling_factor, then checks structural validity.
    
    Designed to be run in parallel with ProcessPool.
    
    Args:
        args: Tuple containing:
            - sampled_prim_slab_coords: Sampled prim slab coordinates (n_samples, n_atoms, 3)
            - sampled_ads_coords: Sampled adsorbate coordinates (n_samples, n_ads_atoms, 3)
            - sampled_lattices: Sampled lattice parameters (n_samples, 6)
            - sampled_supercell_matrices: Supercell transformation matrices (n_samples, 3, 3) or (n_samples, 9)
            - sampled_scaling_factors: Z-direction scaling factors (n_samples,)
            - prim_slab_atom_types: Atomic numbers for prim slab atoms
            - ads_atom_types: Atomic numbers for adsorbate atoms
            - prim_slab_atom_mask: Boolean mask for valid prim slab atoms
            - ads_atom_mask: Boolean mask for valid adsorbate atoms
    
    Returns:
        List of structural validity booleans for each sample.
    """
    (
        sampled_coords,
        sampled_lattices,
        atom_types,
        atom_mask,
        matcher_kwargs,
    ) = args
    
    n_samples = sampled_coords.shape[0]
    valid_atom_types = atom_types[atom_mask]
    results = []
    for i in range(n_samples):
        try:
            # Use assemble function to reconstruct the system
            sampled_lattice = Lattice.from_parameters(
                a=sampled_lattices[i, 0],
                b=sampled_lattices[i, 1],
                c=sampled_lattices[i, 2],
                alpha=sampled_lattices[i, 3],
                beta=sampled_lattices[i, 4],
                gamma=sampled_lattices[i, 5],
            )
            
            sampled_valid_coords = sampled_coords[i][atom_mask]
            
            sampled_structure = Structure(
                lattice=sampled_lattice,
                species=valid_atom_types.tolist(),
                coords=sampled_valid_coords,
                coords_are_cartesian=True,
            )
            # Check structural validity
            struct_valid = _prim_structural_validity(adaptor.get_atoms(sampled_structure))
            results.append(struct_valid)
                
        except Exception as e:
            print(f"WARNING: Failed to compute primitive structural validity for sample {i}: {e}", file=sys.stderr)
            results.append(False)
    
    return results