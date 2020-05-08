import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bootstrap_stat")
    ),
)

import bootstrap_stat
import datasets
