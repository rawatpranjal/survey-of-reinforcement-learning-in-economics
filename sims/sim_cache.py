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


# ---------------------------------------------------------------------------
# Per-component caching API (new, backward-compatible addition)
# ---------------------------------------------------------------------------

def _component_path(cache_dir, script_name, component):
    """Return path: cache_dir/script_name__component.pkl"""
    return os.path.join(cache_dir, f'{script_name}__{component}.pkl')


def save_component(cache_dir, script_name, component, config, data):
    """Save one component's data with its own config hash.

    File: cache_dir/script_name__component.pkl
    """
    os.makedirs(cache_dir, exist_ok=True)
    payload = {'_config_hash': config_hash(config), 'data': data}
    path = _component_path(cache_dir, script_name, component)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"  Cache saved: {path}")


def load_component(cache_dir, script_name, component, config):
    """Load a component if its config hash matches. Returns dict or None."""
    path = _component_path(cache_dir, script_name, component)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('_config_hash') == config_hash(config):
            return payload['data']
    except (pickle.UnpicklingError, EOFError, KeyError):
        pass
    return None


def compute_or_load(cache_dir, script_name, component, config, fn,
                    *args, force=False, **kwargs):
    """Load component from cache, or call fn(*args, **kwargs) and cache the result.

    Parameters
    ----------
    cache_dir : str
        Directory for pickle files.
    script_name : str
        Base name of the script (used in file naming).
    component : str
        Component name, e.g. 'shared', 'FQI', 'CQL'.
    config : dict
        Config dict for this component (hashed for invalidation).
    fn : callable
        Function to call if cache miss. Must return a dict.
    force : bool
        If True, skip cache and recompute.

    Returns
    -------
    dict
        The component's data (from cache or freshly computed).
    """
    if not force:
        cached = load_component(cache_dir, script_name, component, config)
        if cached is not None:
            print(f"  Cache hit: {component}")
            return cached
    print(f"  Computing: {component}")
    data = fn(*args, **kwargs)
    save_component(cache_dir, script_name, component, config, data)
    return data


def add_component_args(parser):
    """Add --data-only, --plots-only, and --algo flags for per-component scripts."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data-only', action='store_true',
                       help='Run computation only, skip figure/table generation')
    group.add_argument('--plots-only', action='store_true',
                       help='Regenerate figures/tables from cached data (no recomputation)')
    parser.add_argument('--algo', type=str, action='append', default=None,
                        help='Force-recompute specific component(s). '
                             'Repeat for multiple: --algo CQL --algo FQI. '
                             '"--algo shared" cascades to all algorithms.')


def parse_force_set(args):
    """Convert parsed --algo flags to a set of component names to force-recompute.

    Returns
    -------
    set
        Component names to force. Empty set means no forcing.
    """
    if args.algo is None:
        return set()
    return set(args.algo)


def list_cached_components(cache_dir, script_name):
    """Return [(component, hash, mtime)] for cached components of a script.

    Useful for debugging cache state.
    """
    results = []
    if not os.path.isdir(cache_dir):
        return results
    prefix = f'{script_name}__'
    for fname in sorted(os.listdir(cache_dir)):
        if fname.startswith(prefix) and fname.endswith('.pkl'):
            component = fname[len(prefix):-4]
            path = os.path.join(cache_dir, fname)
            try:
                with open(path, 'rb') as f:
                    payload = pickle.load(f)
                h = payload.get('_config_hash', '?')
            except Exception:
                h = 'corrupt'
            mtime = os.path.getmtime(path)
            results.append((component, h, mtime))
    return results
