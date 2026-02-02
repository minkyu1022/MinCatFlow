from __future__ import annotations

import argparse
import ast
import lmdb
import math
import numpy as np
import os
import pandas as pd
import pickle
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_all_data_from_lmdb(lmdb_path: str, index: int) -> dict | None:
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    try:
        with db.begin() as txn:
            key = f"{index}".encode("ascii")
            value = txn.get(key)
            
            if value is None:
                return None
            
            data = pickle.loads(value)
        
        data_dict = {}
        
        try:
            data_dict = data.to_dict()
        except (AttributeError, RuntimeError, TypeError):
            pass
        
        if not data_dict or 'pos' not in data_dict:
            try:
                store = getattr(data, '_store', None)
                if store is not None:
                    for key in store.keys():
                        try:
                            data_dict[key] = store[key]
                        except (RuntimeError, AttributeError, KeyError):
                            pass
            except (AttributeError, RuntimeError):
                pass
        
        for key in ['tags', 'atomic_numbers', 'pos', 'pos_relaxed', 'cell', 'sid', 'y_relaxed']:
            if key not in data_dict:
                try:
                    value = getattr(data, key, None)
                    if value is not None:
                        data_dict[key] = value
                except (RuntimeError, AttributeError):
                    pass
        
        sid = None
        try:
            sid = getattr(data, 'sid', None)
        except (RuntimeError, AttributeError):
            pass
        
        if sid is None:
            try:
                if hasattr(data, '__dict__') and 'sid' in data.__dict__:
                    sid = data.__dict__['sid']
            except Exception:
                pass
        
        if sid is None:
            try:
                if hasattr(data, 'items'):
                    for key, value in data.items():
                        if key == 'sid':
                            sid = value
                            break
            except (RuntimeError, AttributeError, TypeError):
                pass
        
        if sid is None:
            try:
                if hasattr(data, 'keys'):
                    keys = list(data.keys())
                    if 'sid' in keys:
                        sid = data['sid']
            except (RuntimeError, AttributeError, TypeError, KeyError):
                pass
        
        if sid is None:
            try:
                store = getattr(data, '_store', None)
                if store is not None and 'sid' in store.keys():
                    sid = store['sid']
            except (RuntimeError, AttributeError, KeyError):
                pass
        
        if sid is None and 'sid' in data_dict:
            sid = data_dict['sid']
        
        if sid is not None:
            if isinstance(sid, (list, tuple)) and len(sid) > 0:
                sid = sid[0]
            elif isinstance(sid, torch.Tensor):
                sid = sid.item() if sid.numel() == 1 else sid.tolist()[0] if len(sid) > 0 else None
        
        tags = None
        if 'tags' in data_dict:
            tags = data_dict['tags']
        else:
            try:
                tags = getattr(data, 'tags', None)
            except (RuntimeError, AttributeError):
                pass
        
        if tags is None:
            return None
        
        if isinstance(tags, torch.Tensor):
            tags_np = tags.cpu().numpy()
        else:
            tags_np = np.array(tags)
        
        atom_mask = (tags_np == 0) | (tags_np == 1) | (tags_np == 2)
        adsorbate_mask = (tags_np == 2)
        
        if not np.any(atom_mask):
            return None
        
        atomic_numbers = None
        if 'atomic_numbers' in data_dict:
            atomic_numbers = data_dict['atomic_numbers']
        else:
            try:
                atomic_numbers = getattr(data, 'atomic_numbers', None)
            except (RuntimeError, AttributeError):
                pass
        
        if atomic_numbers is None:
            return None
        
        if isinstance(atomic_numbers, torch.Tensor):
            atomic_numbers_np = atomic_numbers.cpu().numpy()
        else:
            atomic_numbers_np = np.array(atomic_numbers)
        
        true_system_atomic_numbers = atomic_numbers_np[atom_mask]
        adsorbate_atomic_numbers = atomic_numbers_np[adsorbate_mask] if np.any(adsorbate_mask) else np.array([])
        
        pos = None
        if 'pos' in data_dict:
            pos = data_dict['pos']
        else:
            try:
                pos = getattr(data, 'pos', None)
            except (RuntimeError, AttributeError):
                pass
        
        if pos is None:
            try:
                if hasattr(data, '__dict__') and 'pos' in data.__dict__:
                    pos = data.__dict__['pos']
            except Exception:
                pass
            
            if pos is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos':
                                pos = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
        
        if pos is None:
            return None
        
        if isinstance(pos, torch.Tensor):
            pos_np = pos.cpu().numpy()
        else:
            pos_np = np.array(pos)
        
        true_system_positions = pos_np[atom_mask]
        
        cell = None
        if 'cell' in data_dict:
            cell = data_dict['cell']
        else:
            try:
                cell = getattr(data, 'cell', None)
            except (RuntimeError, AttributeError):
                pass
        
        if cell is None:
            return None
        
        if isinstance(cell, torch.Tensor):
            cell_np = cell.cpu().numpy()
        else:
            cell_np = np.array(cell)
        
        if cell_np.shape == (1, 3, 3):
            cell_np = cell_np[0]
        elif cell_np.shape != (3, 3):
            return None
        
        true_lattice = cell_np
        
        ads_pos_relaxed = None
        pos_relaxed = None
        
        if 'pos_relaxed' in data_dict:
            pos_relaxed = data_dict['pos_relaxed']
        else:
            try:
                pos_relaxed = getattr(data, 'pos_relaxed', None)
            except (RuntimeError, AttributeError):
                pass
            
            if pos_relaxed is None:
                try:
                    if hasattr(data, '__dict__') and 'pos_relaxed' in data.__dict__:
                        pos_relaxed = data.__dict__['pos_relaxed']
                except Exception:
                    pass
            
            if pos_relaxed is None:
                try:
                    if hasattr(data, 'items'):
                        for key, value in data.items():
                            if key == 'pos_relaxed':
                                pos_relaxed = value
                                break
                except (RuntimeError, AttributeError, TypeError):
                    pass
        
        if pos_relaxed is not None:
            if isinstance(pos_relaxed, torch.Tensor):
                pos_relaxed_np = pos_relaxed.cpu().numpy()
            else:
                pos_relaxed_np = np.array(pos_relaxed)
            
            if np.any(adsorbate_mask):
                ads_pos_relaxed = pos_relaxed_np[adsorbate_mask]
        
        y_relaxed = None
        if 'y_relaxed' in data_dict:
            y_relaxed = data_dict['y_relaxed']
        else:
            try:
                y_relaxed = getattr(data, 'y_relaxed', None)
            except (RuntimeError, AttributeError):
                pass
        
        if y_relaxed is None:
            raise ValueError(f"No 'y_relaxed' attribute found in data at index {index}")
        
        return {
            'sid': sid,
            'true_system_atomic_numbers': true_system_atomic_numbers,
            'true_system_positions': true_system_positions,
            'true_lattice': true_lattice,
            'true_tags': tags_np[atom_mask],
            'adsorbate_atomic_numbers': adsorbate_atomic_numbers,
            'ads_pos_relaxed': ads_pos_relaxed,
            'ref_energy': y_relaxed,
        }
        
    except Exception as e:
        return None
    finally:
        db.close()


def get_slab_params_from_mapping(
    mapping_path: str, sid: int
) -> dict[str, str | tuple | float | bool] | None:
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    target_key = f"random{sid}"

    if target_key not in mapping:
        return None

    value = mapping[target_key]
    if not isinstance(value, dict):
        return None

    return {
        "bulk_mpid": value.get("bulk_mpid"),
        "miller_index": value.get("miller_index"),
        "shift": value.get("shift"),
        "top": value.get("top"),
    }


def load_single_bulk(bulk_src_id):
    try:
        from fairchem.data.oc.core import Bulk
        bulk = Bulk(bulk_src_id_from_db=bulk_src_id)
        return bulk_src_id, bulk.atoms.copy(), None
    except Exception as e:
        return bulk_src_id, None, str(e)


def process_row_for_slab_info(args):
    row_idx, row, bulk_atoms_dict = args
    
    try:
        from fairchem.data.oc.core.slab import standardize_bulk
        from pymatgen.core.surface import SlabGenerator
        
        bulk_src_id = str(row["bulk_src_id"])
        specific_miller_str = str(row["specific_miller"])
        
        specific_miller = ast.literal_eval(specific_miller_str)
        if isinstance(specific_miller, list):
            specific_miller = tuple(specific_miller)
        
        if bulk_src_id not in bulk_atoms_dict:
            return row_idx, False, None, None, None, f"Bulk {bulk_src_id} not found in loaded bulks"
        
        bulk_atoms = bulk_atoms_dict[bulk_src_id]
        
        initial_structure = standardize_bulk(bulk_atoms)
        slab_gen = SlabGenerator(
            initial_structure=initial_structure,
            miller_index=specific_miller,
            min_slab_size=7.0,
            min_vacuum_size=20.0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
            max_normal_search=1,
        )
        
        height = slab_gen._proj_height
        n_layers_slab = math.ceil(slab_gen.min_slab_size / height)
        n_layers_vac = math.ceil(slab_gen.min_vac_size / height)
        
        return row_idx, True, n_layers_slab, n_layers_vac, height, None
        
    except Exception as e:
        return row_idx, False, None, None, None, str(e)


def process_row(args):
    global_idx, lmdb_path, local_idx, mapping_path = args
    
    try:
        lmdb_data = get_all_data_from_lmdb(lmdb_path, local_idx)
        
        if lmdb_data is None:
            return global_idx, False, None, "Failed to extract data from LMDB"
        
        sid = lmdb_data['sid']
        
        slab_params = None
        if sid is not None and mapping_path and os.path.exists(mapping_path):
            slab_params = get_slab_params_from_mapping(mapping_path, sid)
        
        result = {
            'sid': sid,
            'bulk_src_id': slab_params['bulk_mpid'] if slab_params else None,
            'specific_miller': str(slab_params['miller_index']) if slab_params and slab_params.get('miller_index') else None,
            'shift': slab_params['shift'] if slab_params else None,
            'top': slab_params['top'] if slab_params else None,
            'true_system_atomic_numbers': str(lmdb_data['true_system_atomic_numbers'].tolist()),
            'true_system_positions': str(lmdb_data['true_system_positions'].tolist()),
            'true_lattice': str(lmdb_data['true_lattice'].tolist()),
            'true_tags': str(lmdb_data['true_tags'].tolist()),
            'adsorbate_atomic_numbers': str(lmdb_data['adsorbate_atomic_numbers'].tolist()) if len(lmdb_data['adsorbate_atomic_numbers']) > 0 else None,
            'ads_pos_relaxed': str(lmdb_data['ads_pos_relaxed'].tolist()) if lmdb_data['ads_pos_relaxed'] is not None else None,
            'ref_energy': lmdb_data['ref_energy'],
        }
        
        return global_idx, True, result, None
        
    except Exception as e:
        return global_idx, False, None, str(e)


def compute_slab_info(
    csv_path: str,
    start_index: int,
    end_index: int | None,
    num_workers: int,
    save_every: int,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    metadata_df = pd.read_csv(csv_path)
    
    if end_index is None:
        end_index = len(metadata_df)
    
    if end_index > len(metadata_df):
        end_index = len(metadata_df)
    
    if 'n_c' not in metadata_df.columns:
        metadata_df['n_c'] = None
    if 'n_vac' not in metadata_df.columns:
        metadata_df['n_vac'] = None
    if 'height' not in metadata_df.columns:
        metadata_df['height'] = None
    
    rows_to_process = metadata_df.iloc[start_index:end_index]
    
    valid_rows = rows_to_process[
        rows_to_process['bulk_src_id'].notna() & 
        rows_to_process['specific_miller'].notna()
    ]
    
    if len(valid_rows) == 0:
        return
    
    unique_bulk_ids = list(valid_rows["bulk_src_id"].unique())
    
    bulk_atoms_dict = {}
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_single_bulk, unique_bulk_ids, chunksize=10),
            total=len(unique_bulk_ids),
            desc="Loading bulks"
        ))
    
    for bulk_src_id, atoms, error in results:
        if atoms is not None:
            bulk_atoms_dict[bulk_src_id] = atoms
    
    task_args = [
        (row_idx, row, bulk_atoms_dict)
        for row_idx, row in valid_rows.iterrows()
    ]
    
    with Pool(num_workers) as pool:
        with tqdm(total=len(task_args), desc="Computing slab info") as pbar:
            for row_idx, success, n_c, n_vac, height, error in pool.imap(
                process_row_for_slab_info, task_args, chunksize=100
            ):
                if success:
                    metadata_df.at[row_idx, 'n_c'] = n_c
                    metadata_df.at[row_idx, 'n_vac'] = n_vac
                    metadata_df.at[row_idx, 'height'] = height
                
                pbar.update(1)
                
                if save_every > 0 and (pbar.n % save_every == 0):
                    metadata_df.to_csv(csv_path, index=False)
    
    metadata_df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract all data from LMDB and mapping file to create val.csv"
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        required=True,
        help="Path to LMDB file or directory containing LMDB files"
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default=None,
        help="Path to oc20_data_mapping.pkl file (optional)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="val.csv",
        help="Output CSV path (default: val.csv)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive). If None, process all indices in LMDB."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: CPU count"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help="Save CSV every N rows (0 = save only at end)"
    )
    parser.add_argument(
        "--num-indices",
        type=int,
        default=None,
        help="Number of indices in LMDB (if known). If None, will try to count."
    )
    
    args = parser.parse_args()
    
    lmdb_path = args.lmdb_path
    mapping_path = args.mapping_path
    output_csv = args.output_csv
    start_index = args.start
    end_index = args.end
    num_workers = args.num_workers or cpu_count()
    save_every = args.save_every
    num_indices = args.num_indices
    
    if os.path.isdir(lmdb_path):
        lmdb_files = []
        for root, dirs, files in os.walk(lmdb_path):
            if 'data.mdb' in files:
                lmdb_files.append(root)
            for file in files:
                if file.endswith('.lmdb') or file.endswith('.mdb'):
                    full_path = os.path.join(root, file)
                    if os.path.isfile(full_path):
                        lmdb_files.append(full_path)
        
        lmdb_files = sorted(list(set(lmdb_files)))
        
        if not lmdb_files:
            raise ValueError(f"No LMDB files found in directory: {lmdb_path}")
    else:
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")
        lmdb_files = [lmdb_path]
    
    lmdb_info = []
    global_idx = 0
    
    for lmdb_file in lmdb_files:
        db = lmdb.open(
            str(lmdb_file),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with db.begin() as txn:
                cursor = txn.cursor()
                file_num_indices = sum(1 for _ in cursor)
        finally:
            db.close()
        
        lmdb_info.append((lmdb_file, file_num_indices, global_idx))
        global_idx += file_num_indices
    
    total_indices = global_idx
    
    if end_index is None:
        end_index = total_indices
    
    if end_index <= start_index:
        raise ValueError(f"end index({end_index}) must be greater than start index({start_index})")
    
    if end_index > total_indices:
        end_index = total_indices
    
    process_args = []
    for lmdb_file, file_num_indices, file_global_start in lmdb_info:
        file_start = max(0, start_index - file_global_start)
        file_end = min(file_num_indices, end_index - file_global_start)
        
        if file_start < file_end:
            for local_idx in range(file_start, file_end):
                global_idx = file_global_start + local_idx
                process_args.append((global_idx, lmdb_file, local_idx, mapping_path))
    
    columns = [
        'sid', 'bulk_src_id', 'specific_miller', 'shift', 'top',
        'true_system_atomic_numbers', 'true_system_positions', 'true_lattice',
        'true_tags', 'adsorbate_atomic_numbers', 'ads_pos_relaxed', 'ref_energy'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    results_dict = {}
    with Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Extracting data") as pbar:
            for global_idx, success, result, error in pool.imap(process_row, process_args):
                if success and result is not None:
                    results_dict[global_idx] = result
                else:
                    results_dict[global_idx] = {
                        'sid': None,
                        'bulk_src_id': None,
                        'specific_miller': None,
                        'shift': None,
                        'top': None,
                        'true_system_atomic_numbers': None,
                        'true_system_positions': None,
                        'true_lattice': None,
                        'true_tags': None,
                        'adsorbate_atomic_numbers': None,
                        'ads_pos_relaxed': None,
                        'ref_energy': None,
                    }
                
                pbar.update(1)
                
                if save_every > 0 and (pbar.n % save_every == 0):
                    sorted_results = [results_dict[idx] for idx in sorted(results_dict.keys())]
                    temp_df = pd.DataFrame(sorted_results)
                    temp_df.to_csv(output_csv, index=False)
    
    sorted_results = [results_dict[idx] for idx in sorted(results_dict.keys())]
    results_df = pd.DataFrame(sorted_results)
    
    results_df.to_csv(output_csv, index=False)
    
    try:
        compute_slab_info(
            csv_path=output_csv,
            start_index=start_index,
            end_index=end_index,
            num_workers=num_workers,
            save_every=save_every,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
