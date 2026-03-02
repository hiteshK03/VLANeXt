"""
DROID Dataset Diagnostic & Validation Tool
===========================================
Scans the DROID TFDS dataset for missing/corrupted keys and reports issues.
Optionally saves a list of valid trajectory indices for filtered training.

Usage:
    # Full scan — check all trajectories and save valid indices
    python -m src.datasets.droid_fixed --droid_path /path/to/droid/1.0.1

    # Quick scan — check first N trajectories
    python -m src.datasets.droid_fixed --droid_path /path/to/droid/1.0.1 --limit 100

    # Scan with multi-view check (wrist camera)
    python -m src.datasets.droid_fixed --droid_path /path/to/droid/1.0.1 --view_mode multi

    # Only report, don't save valid indices
    python -m src.datasets.droid_fixed --droid_path /path/to/droid/1.0.1 --no_save
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from collections import defaultdict

tf.config.set_visible_devices([], 'GPU')

# ---------------------------------------------------------------------------
# Keys that DroidAct.__iter__ accesses
# ---------------------------------------------------------------------------
REQUIRED_TOP_KEYS = ['steps']
REQUIRED_STEP_KEYS = ['reward', 'action', 'observation', 'action_dict', 'language_instruction']
REQUIRED_OBS_KEYS = ['cartesian_position', 'gripper_position']
REQUIRED_ACTION_DICT_KEYS = ['cartesian_velocity', 'gripper_position']

# Camera keys
CAM_KEY = 'exterior_image_1_left'
WRIST_KEY = 'wrist_image_left'


def _get_nested_keys(d, prefix=""):
    """Recursively list all keys in a nested dict/BatchedTensor."""
    keys = []
    for k in d:
        full = f"{prefix}.{k}" if prefix else k
        keys.append(full)
        try:
            sub = d[k]
            if hasattr(sub, 'keys'):
                keys.extend(_get_nested_keys(sub, prefix=full))
        except Exception:
            pass
    return keys


def check_trajectory(traj_data, traj_id, view_mode="single", verbose=False):
    """
    Validate a single trajectory.

    Returns
    -------
    ok : bool
    issues : list[str]   — human-readable issue descriptions
    """
    issues = []

    # --- batch the steps --------------------------------------------------
    try:
        traj_batch = next(iter(traj_data['steps'].batch(5000)))
    except KeyError as e:
        return False, [f"Missing top-level key: {e}"]
    except Exception as e:
        return False, [f"Failed to batch steps: {e}"]

    # --- reward -----------------------------------------------------------
    try:
        reward = traj_batch['reward']
    except KeyError:
        issues.append("Missing key: reward")

    # --- action -----------------------------------------------------------
    try:
        action = traj_batch['action']
        traj_len = action.shape[0]
        if traj_len == 0:
            issues.append("Empty trajectory (action length = 0)")
    except KeyError:
        issues.append("Missing key: action")
        traj_len = None

    # --- observation block ------------------------------------------------
    try:
        obs = traj_batch['observation']
    except KeyError:
        issues.append("Missing key: observation")
        obs = None

    if obs is not None:
        # Camera
        if CAM_KEY not in obs:
            issues.append(f"Missing observation key: {CAM_KEY}")
        else:
            try:
                img = obs[CAM_KEY].numpy()
                if img.ndim != 4:
                    issues.append(f"{CAM_KEY} unexpected ndim={img.ndim} (expected 4: T,H,W,C)")
            except Exception as e:
                issues.append(f"{CAM_KEY} read error: {e}")

        # Wrist camera (only required for multi-view)
        if view_mode == "multi":
            if WRIST_KEY not in obs:
                issues.append(f"Missing observation key (multi-view): {WRIST_KEY}")
            else:
                try:
                    wrist = obs[WRIST_KEY].numpy()
                    if wrist.ndim != 4:
                        issues.append(f"{WRIST_KEY} unexpected ndim={wrist.ndim}")
                except Exception as e:
                    issues.append(f"{WRIST_KEY} read error: {e}")

        # Proprioception
        for k in REQUIRED_OBS_KEYS:
            if k not in obs:
                issues.append(f"Missing observation key: {k}")
            else:
                try:
                    v = obs[k].numpy()
                    if np.any(np.isnan(v)):
                        issues.append(f"NaN found in observation.{k}")
                    if np.any(np.isinf(v)):
                        issues.append(f"Inf found in observation.{k}")
                except Exception as e:
                    issues.append(f"observation.{k} read error: {e}")

        if verbose and obs is not None:
            available_obs_keys = list(obs.keys()) if hasattr(obs, 'keys') else []
            print(f"  [traj {traj_id}] observation keys: {available_obs_keys}")

    # --- action_dict block ------------------------------------------------
    try:
        action_dict = traj_batch['action_dict']
    except KeyError:
        issues.append("Missing key: action_dict")
        action_dict = None

    if action_dict is not None:
        for k in REQUIRED_ACTION_DICT_KEYS:
            if k not in action_dict:
                issues.append(f"Missing action_dict key: {k}")
            else:
                try:
                    v = action_dict[k].numpy()
                    if np.any(np.isnan(v)):
                        issues.append(f"NaN found in action_dict.{k}")
                    if np.any(np.isinf(v)):
                        issues.append(f"Inf found in action_dict.{k}")
                except Exception as e:
                    issues.append(f"action_dict.{k} read error: {e}")

    # --- language_instruction ---------------------------------------------
    try:
        lang = traj_batch['language_instruction']
        first_instr = lang[0].numpy()
        if isinstance(first_instr, bytes):
            first_instr = first_instr.decode('utf-8')
        if not first_instr or first_instr.strip() == "":
            issues.append("Empty language instruction")
    except KeyError:
        issues.append("Missing key: language_instruction")
    except Exception as e:
        issues.append(f"language_instruction read error: {e}")

    ok = len(issues) == 0
    return ok, issues


def scan_dataset(
    droid_path,
    limit=None,
    view_mode="single",
    verbose=False,
    save_path=None,
    success_only=False,
):
    """
    Scan the full DROID dataset and report per-trajectory diagnostics.

    Parameters
    ----------
    droid_path : str
        Path to the DROID TFDS builder directory (e.g. .../droid/1.0.1).
    limit : int or None
        Max trajectories to scan.
    view_mode : str
        "single" or "multi" — multi also checks wrist camera key.
    verbose : bool
        Print available keys for each trajectory.
    save_path : str or None
        If given, save the valid trajectory indices to this JSON file.
    success_only : bool
        If True, also skip trajectories where reward[-1] != 1.
    """
    builder = tfds.builder_from_directory(builder_dir=droid_path)
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)

    total_in_split = builder.info.splits['train'].num_examples
    scan_count = total_in_split if limit is None else min(limit, total_in_split)
    if limit is not None:
        ds = ds.take(limit)

    valid_ids = []
    bad_ids = []
    issue_summary = defaultdict(int)  # issue_text -> count
    skipped_reward = 0
    skipped_error = 0
    first_keys_printed = False

    print(f"\n{'='*60}")
    print(f"DROID Dataset Scan")
    print(f"  Path       : {droid_path}")
    print(f"  Total trajs: {total_in_split}")
    print(f"  Scanning   : {scan_count}")
    print(f"  View mode  : {view_mode}")
    print(f"  Success-only filter: {success_only}")
    print(f"{'='*60}\n")

    pbar = tqdm(enumerate(ds), total=scan_count, unit="traj", desc="Scanning")

    for traj_id, traj_data in pbar:
        # --- Print the available top-level keys once for reference ---
        if not first_keys_printed and verbose:
            try:
                traj_batch = next(iter(traj_data['steps'].batch(5000)))
                all_keys = _get_nested_keys(traj_batch)
                print(f"\n[Info] Available keys in first trajectory:")
                for k in sorted(all_keys):
                    print(f"  {k}")
                print()
                first_keys_printed = True
            except Exception:
                pass

        try:
            ok, issues = check_trajectory(traj_data, traj_id, view_mode=view_mode, verbose=verbose)
        except tf.errors.DataLossError as e:
            ok = False
            issues = [f"TF DataLossError: {e}"]
        except Exception as e:
            ok = False
            issues = [f"Unexpected error: {e}"]

        # Optional: also filter by reward
        if ok and success_only:
            try:
                traj_batch = next(iter(traj_data['steps'].batch(5000)))
                if traj_batch['reward'][-1].numpy() != 1:
                    skipped_reward += 1
                    pbar.set_postfix({"valid": len(valid_ids), "bad": len(bad_ids), "skip_reward": skipped_reward})
                    continue
            except Exception:
                pass  # already checked above

        if ok:
            valid_ids.append(traj_id)
        else:
            bad_ids.append(traj_id)
            for iss in issues:
                issue_summary[iss] += 1
            if len(bad_ids) <= 50:  # print first 50 bad trajectories in detail
                tqdm.write(f"[BAD] traj_id={traj_id}: {issues}")

        pbar.set_postfix({"valid": len(valid_ids), "bad": len(bad_ids)})

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"  Total scanned     : {len(valid_ids) + len(bad_ids) + skipped_reward}")
    print(f"  Valid trajectories: {len(valid_ids)}")
    print(f"  Bad trajectories  : {len(bad_ids)}")
    if success_only:
        print(f"  Skipped (reward)  : {skipped_reward}")
    print()

    if issue_summary:
        print("Issue breakdown:")
        for iss, cnt in sorted(issue_summary.items(), key=lambda x: -x[1]):
            print(f"  [{cnt:>5d}x] {iss}")
        print()

    if bad_ids:
        show = bad_ids[:100]
        print(f"Bad trajectory IDs (first {len(show)}): {show}")
        print()

    # --- Save valid indices ---
    if save_path is not None:
        result = {
            "droid_path": droid_path,
            "view_mode": view_mode,
            "success_only": success_only,
            "total_scanned": len(valid_ids) + len(bad_ids) + skipped_reward,
            "num_valid": len(valid_ids),
            "num_bad": len(bad_ids),
            "valid_ids": valid_ids,
            "bad_ids": bad_ids,
            "issue_summary": dict(issue_summary),
        }
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {save_path}")

    return valid_ids, bad_ids, dict(issue_summary)


def print_trajectory_keys(droid_path, traj_index=0):
    """
    Print all available nested keys for a specific trajectory.
    Useful for debugging what keys exist vs what DroidAct expects.
    """
    builder = tfds.builder_from_directory(builder_dir=droid_path)
    read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
    ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)

    for i, traj_data in enumerate(ds):
        if i < traj_index:
            continue
        print(f"\n=== Trajectory {i} ===")
        try:
            traj_batch = next(iter(traj_data['steps'].batch(5000)))
        except Exception as e:
            print(f"  Failed to batch steps: {e}")
            return

        def _print_keys(d, indent=0):
            prefix = "  " * indent
            for k in sorted(d.keys()) if hasattr(d, 'keys') else []:
                v = d[k]
                if hasattr(v, 'shape'):
                    print(f"{prefix}{k}: shape={v.shape}, dtype={v.dtype}")
                elif hasattr(v, 'keys'):
                    print(f"{prefix}{k}/ (dict)")
                    _print_keys(v, indent + 1)
                else:
                    print(f"{prefix}{k}: type={type(v).__name__}")

        _print_keys(traj_batch)
        break


# ---------------------------------------------------------------------------
# DroidFixed: A robust IterableDataset that skips bad trajectories gracefully
# ---------------------------------------------------------------------------
import gc
import ctypes
import torch
from torch.utils.data import IterableDataset

try:
    _LIBC = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _LIBC.malloc_trim(0)
except Exception:
    def _malloc_trim():
        pass


class DroidFixed(IterableDataset):
    """
    Drop-in replacement for DroidAct with robust key checking.
    
    Differences from DroidAct:
    - Before accessing any key, checks existence → skips bad trajectories cleanly.
    - Optionally loads a pre-computed valid_ids JSON (from scan_dataset) to skip
      known-bad trajectories instantly without re-checking.
    - Logs skipped trajectories with reasons.
    """
    def __init__(
        self,
        droid_path,
        dataset_name='droid',
        length=None,
        history_len=15,
        future_len=15,
        full_sequence=False,
        input_modality="video",
        view_mode="single",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=30000,
        valid_ids_path=None,
    ):
        super().__init__()
        self.droid_path = droid_path
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.buffer_size = buffer_size

        # Load pre-computed valid IDs if available
        self.valid_ids = None
        if valid_ids_path and os.path.exists(valid_ids_path):
            with open(valid_ids_path, 'r') as f:
                data = json.load(f)
            self.valid_ids = set(data['valid_ids'])
            print(f"[DroidFixed] Loaded {len(self.valid_ids)} valid trajectory IDs from {valid_ids_path}")

    def _validate_trajectory(self, traj_batch, traj_id):
        """
        Check that all required keys exist. Returns (ok, reason).
        """
        # Check top-level keys
        for k in ['reward', 'action', 'observation', 'action_dict', 'language_instruction']:
            if k not in traj_batch:
                return False, f"Missing top-level key: {k}"

        obs = traj_batch['observation']
        cam_key = 'exterior_image_1_left'

        # Check camera key
        if cam_key not in obs:
            return False, f"Missing observation key: {cam_key}"

        # Check proprioception keys
        for k in ['cartesian_position', 'gripper_position']:
            if k not in obs:
                return False, f"Missing observation key: {k}"

        # Check action_dict keys
        action_dict = traj_batch['action_dict']
        for k in ['cartesian_velocity', 'gripper_position']:
            if k not in action_dict:
                return False, f"Missing action_dict key: {k}"

        return True, ""

    def __iter__(self):
        builder = tfds.builder_from_directory(builder_dir=self.droid_path)
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        droid_ds = builder.as_dataset(split='train', shuffle_files=False, read_config=read_config)

        if self.length is not None:
            droid_ds = droid_ds.take(self.length)

        shuffle_buffer = []
        BUFFER_SIZE = self.buffer_size

        cam_key = 'exterior_image_1_left'
        wrist_key = 'wrist_image_left'

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        ds_iterator = droid_ds.shard(num_shards=total_shards, index=shard_index)

        ds_iter = iter(ds_iterator)
        traj_id = -1
        skipped_count = 0
        while True:
            try:
                try:
                    traj_data = next(ds_iter)
                except StopIteration:
                    break
                except tf.errors.DataLossError as e:
                    traj_id += 1
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: DataLossError: {e}")
                    skipped_count += 1
                    continue
                traj_id += 1

                # Fast skip if we have a pre-computed valid ID list
                if self.valid_ids is not None and traj_id not in self.valid_ids:
                    skipped_count += 1
                    continue

                try:
                    traj_batch = next(iter(traj_data['steps'].batch(5000)))
                except Exception as e:
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: Failed to batch steps: {e}")
                    skipped_count += 1
                    continue

                # Reward filter
                try:
                    if traj_batch['reward'][-1].numpy() != 1:
                        del traj_batch
                        continue
                except KeyError:
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: Missing 'reward' key")
                    skipped_count += 1
                    del traj_batch
                    continue

                # Validate all required keys
                ok, reason = self._validate_trajectory(traj_batch, traj_id)
                if not ok:
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: {reason}")
                    skipped_count += 1
                    del traj_batch
                    continue

                traj_len = traj_batch['action'].shape[0]
                if traj_len == 0:
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: Empty trajectory")
                    skipped_count += 1
                    del traj_batch
                    continue

                # --- Extract data (same logic as DroidAct) ---
                images_np = traj_batch['observation'][cam_key].numpy()
                if images_np.dtype != np.uint8:
                    images_np = (images_np * 255).astype(np.uint8)

                wrist_np = None
                if self.view_mode == "multi":
                    obs = traj_batch['observation']
                    if wrist_key in obs:
                        wrist_np = obs[wrist_key].numpy()
                        if wrist_np.dtype != np.uint8:
                            wrist_np = (wrist_np * 255).astype(np.uint8)
                    else:
                        wrist_np = images_np.copy()
                    del obs

                cart_pos = traj_batch['observation']['cartesian_position']
                gripper_pos = traj_batch['observation']['gripper_position']
                proprio_np = tf.concat([cart_pos, gripper_pos], axis=-1).numpy().astype(np.float32)

                cart_vel = traj_batch['action_dict']['cartesian_velocity']
                cart_vel = tf.cast(cart_vel, tf.float32)
                cart_vel = tf.clip_by_value(cart_vel, -1.0, 1.0)

                grip_pos = traj_batch['action_dict']['gripper_position']
                grip_cmd = tf.where(grip_pos > 0.5, 1.0, -1.0)
                grip_cmd = tf.cast(grip_cmd, tf.float32)

                actions_np = tf.concat([cart_vel, grip_cmd], axis=-1).numpy().astype(np.float32)

                try:
                    instruction = traj_batch['language_instruction'][0].numpy().decode('utf-8')
                except Exception:
                    instruction = ""

                del traj_batch, cart_pos, gripper_pos, cart_vel, grip_pos, grip_cmd

                # --- Check for NaN/Inf in extracted arrays ---
                if np.any(np.isnan(proprio_np)) or np.any(np.isinf(proprio_np)):
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: NaN/Inf in proprioception")
                    skipped_count += 1
                    del images_np, actions_np, proprio_np
                    continue
                if np.any(np.isnan(actions_np)) or np.any(np.isinf(actions_np)):
                    print(f"[DroidFixed][Warn] Skipping traj {traj_id}: NaN/Inf in actions")
                    skipped_count += 1
                    del images_np, actions_np, proprio_np
                    continue

                if self.full_sequence:
                    sample_indices = np.arange(traj_len)
                else:
                    num_samples = max(1, int(traj_len / (15 * 5)))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)

                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)

                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    hist_imgs = images_np[hist_indices_obs]
                    hist_imgs_wrist = wrist_np[hist_indices_obs] if wrist_np is not None else None
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])

                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = hist_indices_act[valid_mask]
                        valid_indices = np.clip(valid_indices, 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)

                    fut_acts_np = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        valid_indices_fut = fut_indices[valid_mask_fut]
                        fut_acts_np[valid_mask_fut] = actions_np[valid_indices_fut]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    sample = {
                        'proprioception': hist_proprio,
                        'history_actions': hist_actions,
                        'future_actions': fut_acts,
                        'instruction': instruction,
                    }

                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + self.future_len, traj_len - 1)
                        sample['future_image'] = images_np[target_idx].copy()

                    if self.input_modality == "video":
                        sample['video'] = hist_imgs
                        if self.view_mode == "multi":
                            sample['video_wrist'] = hist_imgs_wrist
                    elif self.input_modality == "image":
                        sample['image'] = images_np[t].copy()
                        if self.view_mode == "multi":
                            sample['image_wrist'] = wrist_np[t].copy() if wrist_np is not None else images_np[t].copy()
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)

                    if len(shuffle_buffer) >= BUFFER_SIZE:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

                del images_np, actions_np, proprio_np
                if wrist_np is not None:
                    del wrist_np

            except Exception as e:
                print(f"[DroidFixed][Warn] Skipping traj {traj_id} due to error: {e}")
                skipped_count += 1
                continue
            finally:
                if traj_id % 50 == 0:
                    gc.collect()
                    _malloc_trim()

        if skipped_count > 0:
            print(f"[DroidFixed] Total skipped trajectories: {skipped_count}")

        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DROID Dataset Diagnostic — scan for missing keys, NaN/Inf, and corrupted trajectories."
    )
    parser.add_argument("--droid_path", type=str, required=True,
                        help="Path to DROID TFDS builder dir (e.g. /data/droid/1.0.1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scan first N trajectories (default: all)")
    parser.add_argument("--view_mode", type=str, default="multi", choices=["single", "multi"],
                        help="Check wrist camera key too (default: multi)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print available keys for each trajectory")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save valid IDs JSON")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Custom save path for valid IDs JSON")
    parser.add_argument("--success_only", action="store_true",
                        help="Also filter out trajectories with reward[-1] != 1")
    parser.add_argument("--print_keys", action="store_true",
                        help="Just print all available keys for the first trajectory and exit")
    parser.add_argument("--print_keys_index", type=int, default=0,
                        help="Trajectory index to print keys for (with --print_keys)")
    args = parser.parse_args()

    if args.print_keys:
        print_trajectory_keys(args.droid_path, traj_index=args.print_keys_index)
    else:
        save_path = args.save_path
        if save_path is None and not args.no_save:
            save_path = os.path.join(os.path.dirname(args.droid_path), "droid_valid_ids.json")

        valid_ids, bad_ids, issue_summary = scan_dataset(
            droid_path=args.droid_path,
            limit=args.limit,
            view_mode=args.view_mode,
            verbose=args.verbose,
            save_path=save_path,
            success_only=args.success_only,
        )
