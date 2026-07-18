import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")  # silence old-gym / numpy2 deprecation noise


def pytest_configure(config):
    # the OPS ranking oracle trains an FQE (~20-40s); marked slow so `-m "not slow"`
    # keeps the fast unit suite fast.
    config.addinivalue_line(
        "markers", "slow: heavy test (trains a model); opt out with -m 'not slow'"
    )
