import pandas as pd
import numpy as np
import ast
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from helpers import calculate_rmsd_pymatgen
from helpers import find_vacuum_axis_ase
from helpers import unwrap_adsorbate_fractional
from tqdm import tqdm
import json
import argparse
from ase.io import write, read
import os
import lmdb
import pickle
from multiprocessing import Pool, cpu_count
from pymatgen.core import Structure, Lattice

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0, help='Start index')
parser.add_argument('--end', type=int, default=None, help='End index (use None for all rows to the end)')
parser.add_argument('--num-workers', type=int, default=None, help='Number of parallel workers. Default: CPU count')
parser.add_argument('--chunksize', type=int, default=100, help='Chunk size for multiprocessing')
parser.add_argument('--split', type=str, default='train', help='Split name')
args = parser.parse_args()

start_index = args.start
end_index = args.end
num_workers = args.num_workers or cpu_count()
chunksize = args.chunksize

output_dir = f"processing_results/{args.split}"
fail_path = output_dir + f"failure/"
err_path = output_dir + f"error/"
succ_path = output_dir + f"success/"

metadata_path = f"metadata/{args.split}_metadata.csv"

print(f"Loading metadata CSV: {metadata_path}")
df = pd.read_csv(metadata_path)

# If end_index is None, process up to the length of df
process_end = end_index if end_index is not None else len(df)

print(f"Processing {process_end - start_index} samples (index {start_index} to {process_end-1})")
print(f"Using {num_workers} workers with chunksize {chunksize}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Create LMDB database for successful samples
lmdb_path = os.path.join(output_dir, f"dataset.lmdb")
succ_db = lmdb.open(
    lmdb_path,
    map_size=1099511627776 * 2,  # 2TB max size
    subdir=False,
    meminit=False,
    map_async=True,
)
succ_db_idx = 0  # Counter for successful samples in LMDB

def process_single_sample(args):
    
    idx, df_row = args
    try:
        # Parse row data
        def parse(val, dtype=None, np_type=None):
            if isinstance(val, str):
                val = ast.literal_eval(val)
            if dtype == 'array':
                return np.array(val, dtype=np_type)
            elif dtype:
                return dtype(val)
            return val

        n_c = parse(df_row['n_c'], int)
        n_vac = parse(df_row['n_vac'], int)
        
        sid = parse(df_row['sid'], int)
        tags = parse(df_row['true_tags'], 'array')
        true_system_atomic_numbers = parse(df_row['true_system_atomic_numbers'], 'array', int)
        true_system_positions = parse(df_row['true_system_positions'], 'array', float)
        true_lattice = parse(df_row['true_lattice'], 'array', float)
        ads_pos_relaxed = parse(df_row['ads_pos_relaxed'], 'array', float)
        ref_energy = parse(df_row['ref_energy'], float)

        true_system = Atoms(numbers=true_system_atomic_numbers,
                            positions=true_system_positions,
                            cell=true_lattice,
                            tags=tags,
                            pbc=True)
        slab_mask = (tags == 0) | (tags == 1)
        true_slab = true_system[slab_mask]

        adaptor = AseAtomsAdaptor()

        n_layers = n_c + n_vac

        tight_slab = true_slab.copy()
        tight_cell = tight_slab.get_cell()
        tight_cell[2] = tight_cell[2] * (n_c / n_layers)
        tight_slab.set_cell(tight_cell)
        tight_slab.center()

        tight_slab_struct = adaptor.get_structure(tight_slab)
        prim_slab_struct = tight_slab_struct.get_primitive_structure(tolerance=0.1)
        
        a, b, c, alpha, beta, gamma = prim_slab_struct.lattice.parameters
        standard_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        
        L_orig = prim_slab_struct.lattice.matrix
        L_std = standard_lattice.matrix
        R = (np.linalg.inv(L_orig) @ L_std).T
                
        rotated_prim_struct = Structure(
            standard_lattice,
            prim_slab_struct.species,
            prim_slab_struct.frac_coords  
        )
        
        rotated_tight_lattice = Lattice(tight_slab_struct.lattice.matrix @ R.T)
        
        sc_matrix = np.round(np.dot(
            rotated_tight_lattice.matrix,
            np.linalg.inv(standard_lattice.matrix)
        )).astype(int)

        recon_struct = rotated_prim_struct.copy()
        recon_struct.make_supercell(sc_matrix, to_unit_cell=False)

        recon_tight_slab = adaptor.get_atoms(recon_struct)

        rmsd_tight, _ = calculate_rmsd_pymatgen(
            struct1=recon_tight_slab,
            struct2=tight_slab,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )

        scaling_factor = n_layers / n_c

        recon_slab = recon_tight_slab.copy()
        recon_cell = recon_slab.get_cell()
        recon_cell[2] = recon_cell[2] * scaling_factor
        recon_slab.set_cell(recon_cell)

        rmsd_w_vac, _ = calculate_rmsd_pymatgen(
            struct1=recon_slab,
            struct2=true_slab,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )
        
        rotated_true_slab = true_slab.copy()
        rotated_true_slab.set_cell(true_slab.cell @ R.T)
        rotated_true_slab.set_positions(true_slab.positions @ R.T)

        adsorbate = true_system[tags == 2].copy()
        
        rotated_adsorbate = adsorbate.copy()
        rotated_adsorbate.set_positions(adsorbate.positions @ R.T)
        diff_vector = recon_slab.get_center_of_mass() - rotated_true_slab.get_center_of_mass()
        rotated_adsorbate.translate(diff_vector)

        rmsd_system, _ = calculate_rmsd_pymatgen(
            struct1=recon_slab + rotated_adsorbate,
            struct2=true_system,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )

        relaxed_adsorbate = adsorbate.copy()
        relaxed_adsorbate.set_positions(ads_pos_relaxed @ R.T)
        diff_vector = recon_slab.get_center_of_mass() - rotated_true_slab.get_center_of_mass()
        relaxed_adsorbate.translate(diff_vector)
        
        adsorbate = relaxed_adsorbate

        # Check if all RMSD values are valid and below threshold
        if (rmsd_tight is not None and rmsd_w_vac is not None and rmsd_system is not None and
            rmsd_tight < 1e-4 and rmsd_w_vac < 1e-4 and rmsd_system < 1e-4):

            # Convert primitive slab from pymatgen Structure to ASE Atoms
            primitive_slab_atoms = adaptor.get_atoms(rotated_prim_struct)

            # Prepare data dictionary
            dataset_dict = {
                "sid": sid,
                "primitive_slab": primitive_slab_atoms,
                "supercell_matrix": sc_matrix,
                "n_slab": n_c,
                "n_vac": n_vac,
                "ads_atomic_numbers": adsorbate.numbers,
                "ads_pos": adsorbate.positions,
                "ref_energy": ref_energy
            }

            return {
                'idx': idx,
                'status': 'success',
                'rmsd': (rmsd_tight, rmsd_w_vac),
                'dataset_dict': dataset_dict
            }
        else:
            return {
                'idx': idx,
                'status': 'fail',
                'rmsd': (rmsd_tight, rmsd_w_vac),
                'dataset_dict': None
            }

    except Exception as e:
        return {
            'idx': idx,
            'status': 'error',
            'error': str(e),
            'rmsd': None,
            'dataset_dict': None
        }


# Prepare arguments for multiprocessing
rows_to_process = df.iloc[start_index:process_end]
task_args = [
    (start_index + i, row)
    for i, (_, row) in enumerate(rows_to_process.iterrows())
]

succ_samples = {}
fail_samples = {}
err_samples = {}

# Process samples in parallel
print("Processing samples in parallel...")
with Pool(num_workers) as pool:
    results = list(tqdm(
        pool.imap(process_single_sample, task_args, chunksize=chunksize),
        total=len(task_args),
        desc="Processing"
    ))

# Collect results and save to LMDB
print("Collecting results and saving to LMDB...")
for result in tqdm(results, desc="Saving"):
    idx = result['idx']
    
    if result['status'] == 'success':
        succ_samples[idx] = result['rmsd']
        
        # Save to LMDB
        with succ_db.begin(write=True) as txn:
            txn.put(
                f"{succ_db_idx}".encode("ascii"),
                pickle.dumps(result['dataset_dict'], protocol=-1)
            )
        succ_db_idx += 1
        
    elif result['status'] == 'fail':
        fail_samples[idx] = result['rmsd']
        
    elif result['status'] == 'error':
        err_samples[idx] = result['error']

# Save count of successful samples in LMDB
with succ_db.begin(write=True) as txn:
    txn.put("length".encode("ascii"), pickle.dumps(succ_db_idx, protocol=-1))
succ_db.sync()
succ_db.close()

# Save the dictionaries to JSON files
print("\nSaving results to JSON files...")

with open(os.path.join(output_dir, "succ_samples.json"), "w") as f:
    json.dump(succ_samples, f, indent=4)

with open(os.path.join(output_dir, "fail_samples.json"), "w") as f:
    json.dump(fail_samples, f, indent=4)

with open(os.path.join(output_dir, "err_samples.json"), "w") as f:
    json.dump(err_samples, f, indent=4)

print(f"\nSuccessfully saved results in '{output_dir}'")
print(f"LMDB database: {lmdb_path}")
print(f"Total processed: {len(results)}")
print(f"Successful samples: {len(succ_samples)}")
print(f"Failed samples: {len(fail_samples)}")
print(f"Errored samples: {len(err_samples)}")