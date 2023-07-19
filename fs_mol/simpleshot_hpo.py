"""
Launches Simpleshot on a given fold, to measure the performance of the model.
"""
import sys
import argparse
import logging
import time
from dataclasses import asdict

from pyprojroot import here as project_root
import wandb

sys.path.insert(0, str(project_root()))

from fs_mol.simpleshot_test import test
from fs_mol.utils.simpleshot_utils import SimpleShotConfig


logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

wandb.login()


def parse_command_line():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launches Simpleshot on a given fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run on wandb",
    )
    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION"],
        default="VALIDATION",
        help="Fold to launch the test on",
    )

    parser.add_argument(
        "--descriptor",
        type=str,
        default="ecfp",
        help="Descriptor to use for the model",
    )

    parser.add_argument(
        "--support_set_size",
        nargs="+",
        default=[16],
        help="Size of support set",
    )

    args = parser.parse_args()
    return args


def run_hpo_factory(args, name):
    """Run HPO on a given fold"""

    def run_hpo_fn(name=name):
        """Run HPO on a given fold"""
        wandb.init(
            project="FS-Mol",
            name=name + f"""_{time.strftime("%d_%b_%H_%M_%S", time.localtime())}""",
        )
        name_run = name + f"""_{wandb.config["lmbd_entropy"]}"""
        name_run += f"""_{wandb.config["temperature"]}"""
        name_run += f"""_{wandb.config["learning_rate"]}"""
        name_run += f"""_{wandb.config["epochs"]}"""
        name_run += f"""_{wandb.config["clip_grad_norm"]}"""
        name_run += f"""_{wandb.config["center_data"]}"""
        name_run += f"""_{wandb.config["normalize_norm"]}"""

        model_config = SimpleShotConfig(
            lmbd_entropy=wandb.config["lmbd_entropy"],
            temperature=wandb.config["temperature"],
            learning_rate=wandb.config["learning_rate"],
            epochs=wandb.config["epochs"],
            clip_grad_norm=wandb.config["clip_grad_norm"],
            center_data=wandb.config["center_data"],
            normalize_norm=wandb.config["normalize_norm"],
            distance=wandb.config["distance"],
            bias=wandb.config["bias"],
        )
        test(
            model_config=model_config,
            descriptor=args.descriptor,
            fold=args.fold,
            name=name_run,
            support_set_size=args.support_set_size,
            out_file="",
            wandb_init=False,
        )

    return run_hpo_fn


if __name__ == "__main__":
    args = parse_command_line()

    if args.name == "":
        name = "Simpleshot_HPO_"
    else:
        name = args.name

    sweep_config = {
        "method": "random",
        "metric": {"name": "delta_aucpr_16", "goal": "maximize"},
        "parameters": {
            "lmbd_entropy": {"min": 0.0, "max": 0.5},
            "temperature": {"min": 0.1, "max": 3.0},
            "learning_rate": {"values": [1e-3]},
            "epochs": {"values": [10]},
            "clip_grad_norm": {"values": [1.0]},
            "center_data": {"values": [True, False]},
            "normalize_norm": {"values": [True]},
            "distance": {"values": ["cosine"]},
            "bias": {"values": [True, False]},
            "out_shape":{"values": [16,32,64,128]},
        },
    }

    hpo_fn = run_hpo_factory(args, name)

    sweep_id = wandb.sweep(sweep=sweep_config, project="FS-Mol")

    wandb.agent(sweep_id, function=hpo_fn, count=200)
