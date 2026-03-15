# Shared simulation caching module.
# Provides config hashing, save/load, and argparse flags for --data-only / --plots-only.

import argparse
import hashlib
import json
import os
import pickle


def config_hash(config: dict) -> str:
    """MD5 hash of JSON-serialized config dict for cache invalidation."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def save_results(cache_dir: str, name: str, config: dict, data: dict):
    """Pickle data with config hash to cache_dir/name.pkl."""
    os.makedirs(cache_dir, exist_ok=True)
    data['_config_hash'] = config_hash(config)
    path = os.path.join(cache_dir, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Cache saved: {path}")


def load_results(cache_dir: str, name: str, config: dict):
    """Load cached data if config hash matches. Returns dict or None."""
    path = os.path.join(cache_dir, f'{name}.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if data.get('_config_hash') == config_hash(config):
            return data
    except (pickle.UnpicklingError, EOFError, KeyError):
        pass
    return None


def add_cache_args(parser: argparse.ArgumentParser):
    """Add mutually exclusive --data-only and --plots-only flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data-only', action='store_true',
                       help='Run computation only, skip figure/table generation')
    group.add_argument('--plots-only', action='store_true',
                       help='Regenerate figures/tables from cached data (no recomputation)')
