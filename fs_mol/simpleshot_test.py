"""
Launches Simpleshot on a given fold, to measure the performance of the model.
"""
import os
import sys
import argparse
import logging
import json
import time
from functools import partial
from dataclasses import asdict

import pandas as pd
from pyprojroot import here as project_root
from tqdm import tqdm
import wandb

sys.path.insert(0, str(project_root()))

from fs_mol.models.simpleshot import SimpleShotConfig
from fs_mol.utils.simpleshot_utils import (
    test_model_fn,
)
from fs_mol.data import FSMolDataset, DataFold
from fs_mol.utils.test_utils import eval_model

BASE_SUPPORT_SET_SIZE = [16, 32, 64, 128]

logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

FS_MOL_CHECKOUT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
FS_MOL_DATASET_PATH = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets/fs-mol"
)
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)


def update_wandb_config(config, name):
    """Update wandb config"""

    wandb.config.update({f"{name}": asdict(config)})


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
        "--model_config",
        type=str,
        default="fs_mol/configs/SimpleshotConfig/model_config.json",
        help="Path to the model config file",
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION", "TEST"],
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
        default=[16, 32, 64],
        help="Size of support set",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default="results_simpleshot.csv",
        help="Path to the output file",
    )

    args = parser.parse_args()
    return args


def test(
    name, fold, descriptor, support_set_size, out_file, model_config, wandb_init=True
):
    wandb.init(
        project="FS-Mol",
        name=name,
    )


    def update_wandb_config(config, name):
        """Update wandb config"""

        wandb.config.update({f"{name}": asdict(config)})

    wandb.config.update({"descriptor": descriptor})
    wandb.config.update({"support_set_size": support_set_size})

    update_wandb_config(model_config, "model_config")

    logger.info(f"Launching Simpleshot on fold {fold}")
    logger.info(f"Model config: {model_config}")

    FOLD = DataFold[fold]
    fsmol_dataset = FSMolDataset.from_directory(FS_MOL_DATASET_PATH)

    p_bar = tqdm(
        total=len(support_set_size) * len(fsmol_dataset._fold_to_data_paths[FOLD]),
        desc="Progression : ",
        leave=True,
    )
    for i in BASE_SUPPORT_SET_SIZE:
        wandb.define_metric(f"delta_aucpr_{i}", hidden=True)
    full_results = pd.DataFrame()
    for support_size in support_set_size:
        logger.info(f"Support set size: {support_size}")
        results = eval_model(
            test_model_fn=partial(
                test_model_fn,
                descriptor=descriptor,
                model_config=model_config,
                p_bar=p_bar,
            ),
            dataset=fsmol_dataset,
            fold=FOLD,
            train_set_sample_sizes=[support_size],
            num_samples=1,
        )

        results = {k: asdict(v[0]) for k, v in results.items()}
        df_task_result = pd.DataFrame(results).T
        df_task_result["delta_aucpr"] = (
            df_task_result["avg_precision"] - df_task_result["fraction_pos_test"]
        )
        results_wandb_table = wandb.Table(
            dataframe=df_task_result[["delta_aucpr", "fraction_pos_test"]]
        )
        wandb.log(
            {
                "Frac_test_plot": wandb.plot.scatter(
                    results_wandb_table,
                    "fraction_pos_test",
                    "delta_aucpr",
                    title="frac_pos_test",
                )
            }
        )
        df_task_result["support_set_size"] = support_size
        full_results = pd.concat([full_results, df_task_result])
        if support_size in BASE_SUPPORT_SET_SIZE:
            wandb.log(
                {f"delta_aucpr_{support_size}": df_task_result.delta_aucpr.mean()}
            )

    # wandb.log({"Results_table": wandb.Table(dataframe=df_task_result)})
    res_grouped = (
        full_results.groupby("support_set_size")["delta_aucpr"].mean().reset_index()
    )
    res_table = wandb.Table(dataframe=res_grouped)

    wandb.log(
        {
            "Results": wandb.plot.line(
                res_table,
                "support_set_size",
                "delta_aucpr",
                title="Results",
            )
        }
    )
    if not out_file == "":
        full_results.to_csv(out_file)
    wandb.finish()


if __name__ == "__main__":
    args = parse_command_line()
    name, fold, descriptor, support_set_size, out_file = (
        args.name,
        args.fold,
        args.descriptor,
        args.support_set_size,
        args.out_file,
    )
    name = (
        name
        if not name == ""
        else f"""Simpleshot_{time.strftime("%d_%b_%H_%M", time.localtime())}"""
    )

    wandb.login()

    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    model_config = SimpleShotConfig(**model_config)
    test(name, fold, descriptor, support_set_size, out_file, model_config)
