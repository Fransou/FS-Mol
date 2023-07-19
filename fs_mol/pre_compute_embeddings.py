import os
import sys
import numpy as np
import argparse
import time

import datamol as dm
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.base import MoleculeTransformer

from molfeat.calc.pharmacophore import Pharmacophore3D
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_command_line():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Computes embeddings for a given fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION", "TEST"],
        default="VALIDATION",
        help="Fold to launch the test on",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default="results_simpleshot.csv",
        help="Path to the output file",
    )

    args = parser.parse_args()
    return args


args = parse_command_line()


# This should be the location of the checkout of the FS-Mol repository:
FS_MOL_CHECKOUT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
FS_MOL_DATASET_PATH = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "datasets/fs-mol"
)
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.data import FSMolDataset, DataFold

dataset = FSMolDataset.from_directory(FS_MOL_DATASET_PATH)

fold = DataFold[args.fold]
task_iterable = dataset.get_task_reading_iterable(fold)  # FOLD to featurize
task = next(iter(task_iterable))

df = {"smiles": []}


def add_features_molecule(smiles, df):
    """Modify this cell to change the computation of features for each molecule."""
    transformer = FPVecTransformer(kind="electroshape", dtype=float)
    mol = dm.to_mol(smiles)
    try:
        mol = dm.conformers.generate(mol, align_conformers=True, n_confs=5)
        features = transformer(mol)
    except:
        features = np.zeros((1, 2048))
    for i_feature in range(len(features)):
        for i in range(len(features[i_feature])):
            if f"embedding_{i}" not in df.keys():
                df[f"embedding_{i}"] = []
            df[f"embedding_{i}"].append(features[i_feature, i])
    df["smiles"].append(smiles)


df_base = pd.DataFrame(
    {
        "smiles": [s.smiles for s in task.samples],
        "label": [int(s.bool_label) for s in task.samples],
    }
)


_ = df_base.progress_apply(lambda x: add_features_molecule(x["smiles"], df), axis=1)
df = pd.DataFrame(df).join(df_base.set_index("smiles"), on="smiles")

embeddings_keys = [k for k in df.keys() if k.startswith("embedding_")]

X = df[embeddings_keys].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = df["label"].values


try:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(
        X,
    )
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = y

    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label")
except Exception as e:
    print(e)
    print("PCA failed, skipping plot.")

df_tasks = {"smiles": []}
for task in tqdm(iter(task_iterable)):
    df_tasks = {"smiles": [s.smiles for s in task.samples]}
df_tasks = pd.DataFrame(df_tasks)

df_embeddings = {"smiles": []}
_ = df_tasks.progress_apply(
    lambda x: add_features_molecule(x["smiles"], df_embeddings), axis=1
)
df_embeddings = pd.DataFrame(df_embeddings).join(df_tasks.set_index("smiles"), on="smiles")


name = args.out_file
name = f"""Embeddings_{time.strftime("%d_%b_%H_%M_%S", time.localtime())}""" if name=="" else name

df_embeddings.to_csv(name)

