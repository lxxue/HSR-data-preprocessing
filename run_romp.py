"""
ROMP Execution Script

Arguments:
    --data_dir: path to data directory containing images
    --gender: gender for SMPL model (male/female)
    --romp_ckpt_dir: path to ROMP checkpoint directory (default: checkpoints/romp/)

Output Structure:
    data_dir/
    └── smpl/
        └── ROMP/              # SMPL parameters (npz) and visualization (png) 
"""

import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--gender", type=str, choices=["male", "female"], required=True)
    parser.add_argument("--romp_ckpt_dir", type=Path, default="checkpoints/romp/")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    gender = args.gender
    romp_ckpt_dir = args.romp_ckpt_dir.resolve()

    cmd = [
        "romp",
        "--mode",
        "video",
        "--calc_smpl",
        "--render_mesh",
        "-i",
        str(data_dir / "images" / "selected_frames"),
        "-o",
        str(data_dir / "smpl" / "ROMP"),
        "--smpl_path",
        Path(romp_ckpt_dir / f"SMPL_{gender.upper()}.pth").resolve(),
    ]
    subprocess.run(["echo"] + cmd)
    subprocess.run(cmd)
