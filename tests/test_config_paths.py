from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deskpilot.core.config import DATA_DIR, ROOT_DIR as CONFIG_ROOT_DIR


def test_config_root_dir_points_to_project_root() -> None:
    assert CONFIG_ROOT_DIR == ROOT_DIR
    assert DATA_DIR == ROOT_DIR / "data"


def main() -> None:
    test_config_root_dir_points_to_project_root()
    print("Config path tests passed.")


if __name__ == "__main__":
    main()
