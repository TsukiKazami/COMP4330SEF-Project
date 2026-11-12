"""Run the comp4330 CLI without installing the package.

    PYTHONPATH=src python scripts/colab_cli.py rq1 --gt ...
"""

import sys

from comp4330sef.cli import main

raise SystemExit(main(sys.argv[1:]))