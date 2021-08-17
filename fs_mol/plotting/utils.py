""" A collection of utils to collate model evaluation data, and plot results"""
import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from typing import Tuple, List, Optional, Callable, Dict, Iterable, Union

from pandas.core.base import DataError
from azureml.core import Run


def default_taskname_extractor_fn(filename: str) -> str:
    filename = os.path.basename(filename)

    name_components = re.split("_|-", filename)

    for component in name_components:
        if component.startswith("CHEMBL"):
            return component

    if len(name_components) > 1:
        return name_components[1]

    raise ValueError("Do not know how to extract task name from filename!")


def summarize_test_run(
    df, grouping_column: str = "train_frac", control_col_strings: str = "num|_frac|seed"
) -> Tuple[int, pd.DataFrame]:
    """Takes a csv output from running finetuning_eval.py on one test dataset,
    and summarises data; means and standard deviations across runs with the same
    value of "grouping_column" are found. Returned as collapsed summary dataframe.

    Args:
        df (pd.DataFrame): Input dataframe of single test task results across
        multiple runs (different seeds, train set sizes).
        grouping_column (str): Column name by which the data should be grouped.
        control_col_strings (str): Identifiers for all columns that were control variables.

    Returns:
        (int, pd.DataFrame): Total number of datapoints in train and test for this run,
        summarised results dataframe.
    """
    measurement_cols = list(df.columns[~df.columns.str.contains(control_col_strings)])
    # control_cols = list(df.columns[df.columns.str.contains(control_col_strings)]).remove(
    #     grouping_column
    # )

    sdf = df.groupby([grouping_column]).mean().drop(columns=["seed"])
    rdf = (
        df.groupby([grouping_column])
        .std()
        .drop(columns=["seed"])
        .rename(columns={f"{x}": f"{x}_std" for x in measurement_cols})
    )

    total_datapoints = df["num_train"].max() + df["num_test"].min()

    cols_to_use = rdf.columns.difference(sdf.columns)

    return (
        total_datapoints,
        sdf.merge(rdf[cols_to_use], how="inner", left_index=True, right_index=True),
    )


def plot_test_run(
    task: str,
    model_name: str,
    results_df: pd.DataFrame,
    grouping_column: str = "num_train_requested",
    plotting_metric: str = "average_precision_score",
    output_dir: str = "outputs",
) -> plt.figure:
    """Plot the summary of results for a single test run.

    Args:
        task (str): name of test task
        results_df (pd.DataFrame): Summary dataframe for single task run
        grouping_column (str): control variable to plot.
        plotting_metric (str): performance metric to be plotted.
        output_dir (str): output directory to save plots
        total_datapoints (int): total datapoints in this task

    Returns:
        None
    """
    total_datapoints = results_df["num_train"].max() + results_df["num_test"].min()
    x_pos = np.arange(len(results_df))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(
        x_pos,
        results_df[plotting_metric],
        yerr=results_df[f"{plotting_metric}_std"],
        alpha=0.7,
        ecolor="black",
        capsize=8,
        fmt="o",
        ms=15,
    )
    ax.set_ylabel(plotting_metric, fontsize=20)
    ax.set_xlabel(grouping_column, fontsize=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df[grouping_column].values, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_title(f"Performance on assay {task}", fontsize=22)
    ax.yaxis.grid(True)

    def tick_function(x):
        # convert fraction to training points for this df
        return [int(i * total_datapoints) for i in x]

    if grouping_column.__contains__("frac"):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(tick_function(results_df[grouping_column]))
        ax2.set_xlabel("Number of training data points")

    plt.savefig(os.path.join(output_dir, f"{model_name}_{task}_plot.png"))
    # plt.close()
    return fig


def get_csv_paths(
    input_dir: str = "results",
    files_suffix: str = "",
    files_prefix: str = "",
) -> List[str]:

    filestr = f"{files_prefix}*{files_suffix}.csv"
    csv_summary_files = sorted(glob(os.path.join(input_dir, filestr)))
    return csv_summary_files


def get_amlrun_csvs(
    run,
    output_dir: str = "results",
    task_name_extraction_fn: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Get complete paths to results csvs: each run is assumed to be for a single test task.
    If mount_path is a valid path to a mounted drive, returns paths to csvs on mounted drive
    If mount path is None or invalid, finds run csvs from run object, downloads to the results directory,
    and returns full paths.

    run: azureml ScriptRun object
    output_dir: str
    """

    if task_name_extraction_fn is not None:
        _task_name_extraction_fn = task_name_extraction_fn
    else:
        _task_name_extraction_fn = default_taskname_extractor_fn

    run_files = run.get_file_names()
    run_files_outputs_csvs = [
        x for x in run_files if x.startswith("outputs") and x.endswith(".csv")
    ]

    run_csv_summary_file = []

    # if there is more than one file put them into a separate directory.
    if len(run_files_outputs_csvs) > 1:
        for i, filename in enumerate(run_files_outputs_csvs):
            # name for csv file without "outputs" from aml
            basename = filename.split("/")[1]
            task_name = _task_name_extraction_fn(basename)
            output_path = os.path.join(output_dir, f"downloaded_{task_name}/{basename}")
            print(f"downloading file {filename}, saving to {output_dir}.")
            run.download_file(filename, output_file_path=output_path)
            if i == 0:
                df = pd.read_csv(output_path)
            else:
                df = pd.concat([df, pd.read_csv(output_path)], axis=0, ignore_index=True)
        final_output_path = os.path.join(output_dir, basename)
        df.to_csv(final_output_path, index=False, header=True)
        run_csv_summary_file.append(final_output_path)
    else:
        # otherwise just download and append file name to return list
        filename = run_files_outputs_csvs[0]
        basename = filename.split("/")[1]
        task_name = _task_name_extraction_fn(basename)
        output_path = os.path.join(output_dir, basename)
        print(f"downloading file {filename}, saving to {output_dir}.")
        run.download_file(filename, output_file_path=output_path)
        run_csv_summary_file.append(output_path)

    return run_csv_summary_file


def process_file(
    summary_dfs: Dict[int, pd.DataFrame],
    file: str,
    output_dir: str,
    model_name: str,
    select_values: Iterable[int],
    task_name_extraction_fn: Optional[Callable[[str], str]] = None,
    grouping_column: str = "num_train_requested",
    plot: bool = False,
    plotting_metric: str = "roc_auc",
) -> Dict[int, pd.DataFrame]:
    print(f"processing {file}")
    if task_name_extraction_fn is not None:
        _task_name_extraction_fn = task_name_extraction_fn
    else:
        _task_name_extraction_fn = default_taskname_extractor_fn

    df = pd.read_csv(file)
    try:
        _, results_df = summarize_test_run(df, grouping_column=grouping_column)
        assay = _task_name_extraction_fn(file)
        savename = f"{model_name}_{assay}.csv"
        results_df.to_csv(os.path.join(output_dir, savename), header=True, index=True)
        results_df.reset_index(inplace=True)
        results_df["assay"] = pd.Series([assay for x in range(len(df.index))])

        if plot:
            fig = plot_test_run(
                assay,
                model_name=model_name,
                results_df=results_df,
                grouping_column="num_train",  # use actual num train in contrast to grouping by "requested"
                plotting_metric=plotting_metric,
                output_dir=output_dir,
            )
            plt.close(fig)

        # loop over the number of train_requested points we want to group by
        for val in select_values:
            if summary_dfs[val].empty:
                summary_dfs[val] = pd.DataFrame(columns=results_df.columns)
            summary_dfs[val] = summary_dfs[val].append(
                results_df.loc[results_df[grouping_column] == val], ignore_index=True
            )

        print(f"Summarised results for test on assay {assay}")
    except DataError as e:
        print(f"Failed to process {file}, likely empty: {e}")

    return summary_dfs


def collate_experiment_results(
    results_dir: str = "outputs",
    model_name: str = "model",
    x_col: str = "num_train_requested",
    y_col: str = "average_precision_score",
    task_col: str = "assay",
    task_number_prefix: str = "CHEMBL",
    num_points: List[int] = [16, 32, 64, 128, 256],
) -> pd.DataFrame:

    # method to join together all the results for a particular experiment,
    # across all number of training points
    df_list = []
    # these are here for back compatibility
    frac_cols = [
        "fraction_positive_train",
        "fraction_positive_test",
        "fraction_positive_train_std",
        "fraction_positive_test_std",
    ]

    for num in num_points:
        try:
            df = pd.read_csv(os.path.join(results_dir, f"{model_name}_summary_{x_col}_{num}.csv"))
        except Exception as e:
            print(f"No file for {num}: + {e}")
            continue

        if len(df) > 0:
            df = _clean_assay(
                df,
                num,
                x_col,
                y_col,
                task_col,
                task_number_prefix,
                drop_frac_col=False,
                frac_cols=frac_cols,
            )
            df_list.append(df)

    merge_cols = ["TASK_ID"]
    included_frac_cols = 0
    for col in frac_cols:
        if col in df_list[0].columns.values:
            merge_cols.append(col)
            included_frac_cols += 1

    merged_df = df_list[0]
    for df in df_list[1:]:
        # note this requires a master df to conform to this format of columns
        # TODO: make more general
        merged_df = merged_df.merge(df, how="outer", on=merge_cols)

    # clean up to account for the variation in fraction of positive points in train/test
    if included_frac_cols > 0:
        rdf = (
            merged_df.drop(columns=frac_cols)
            .groupby("TASK_ID")
            .agg(lambda x: np.nan if x.isnull().all() else x.dropna())
            .reset_index()
        )
        merged_df = merged_df.groupby(["TASK_ID"]).mean().reset_index().merge(rdf, on="TASK_ID")

    return merged_df


def _clean_assay(
    df: pd.DataFrame,
    num_points: int,
    x_col: str,
    y_col: str,
    task_col: str,
    task_number_prefix: Optional[str] = None,
    drop_frac_col: bool = True,
    frac_cols: List[str] = [
        "fraction_positive_train",
        "fraction_positive_test",
        "fraction_positive_train_std",
        "fraction_positive_test_std",
    ],
) -> pd.DataFrame:
    keep_cols = [y_col, f"{y_col}_std", x_col, task_col]

    # make back compatible with previous output of evaluation where these columns may not exist
    for col in frac_cols:
        if col in df.columns.values:
            keep_cols.append(col)

    # remove unwanted measurement columns
    to_drop = list(set(df.columns.values) - set(keep_cols))
    df.drop(columns=to_drop, inplace=True)

    # listing task ids by number if they have a prefix, or just task name
    if task_number_prefix is None:
        df["TASK_ID"] = df[task_col]
    else:
        df["TASK_ID"] = df.apply(lambda x: int(x[task_col][len(task_number_prefix) :]), axis=1)

    # model name and number of points, so that final result can be concatenated with
    # other model results later
    df[f"{num_points}_train"] = (
        df[y_col]
        .round(decimals=3)
        .astype(str)
        .str.cat(df[f"{y_col}_std"].round(decimals=3).astype(str), sep="+/-")
    )

    # clean up dataframe to prevent proliferation of columns when merging over
    # num_train_requested versions.
    to_drop = [f"{y_col}", f"{y_col}_std", task_col, x_col]
    if drop_frac_col:
        # back compat again
        for col in frac_cols:
            if col in df.columns.values:
                to_drop.append(col)

    df = df.drop(columns=to_drop)

    return df


def highlight_max_all(row, sizes_to_compare: List[int] = [16, 32, 64, 128, 256]) -> List[str]:
    """
    highlight the maximum in the passed data for each subset
    """

    # Strip the error from our numbers: +/-
    row = row.apply(lambda x: get_number_from_val_plusminus_error(x))

    # for each subset of cols get the max
    size_to_max = {}
    for size in sizes_to_compare:
        columns_for_size = [col_name for col_name in row.index if col_name.startswith(f"{size}_")]
        size_to_max[size] = row[columns_for_size].max()

    # now walk over the columns and set bold attr if it's the maximum:
    bold_attr = "font-weight: bold; color: green"
    attr_list = []
    for col_name in row.index:
        maybe_size = col_name.split("_")[0]
        try:
            size = int(maybe_size)
            if row[col_name] == size_to_max[size]:
                attr_list.append(bold_attr)
            else:
                attr_list.append("")
        except Exception:
            attr_list.append("")

    return attr_list


def get_number_from_val_plusminus_error(
    x: Union[str, float],
    get_error: bool = False,
) -> float:
    # convert "+/-" values from dataframes to floats
    if not isinstance(x, str):  # or x in ignore_vals:
        return x
    else:
        if get_error:
            index = 1
        else:
            index = 0
        return float(x.split("+/-")[index])


def plot_all_assays(
    df: pd.DataFrame,
    model_names: Iterable[str],
    sizes_to_compare: List[int] = [16, 32, 64, 128, 256],
    results_dir: Optional[str] = None,
):
    plt.rc("font", size=10)  # controls default text size
    plt.rc("axes", titlesize=14)  # fontsize of the title
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=12)  # fontsize of the legend

    for task in df["TASK_ID"]:
        # random classifier performance
        baseline = df.loc[df["TASK_ID"] == task]["fraction_positive_test"].values
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            # we add try except as some tasks failed completely.
            for model_name in model_names:
                cols = [x for x in df.columns if x.endswith(f" ({model_name})")]
                string_values = df.loc[df["TASK_ID"] == task][cols].values
                results = [
                    get_number_from_val_plusminus_error(x, get_error=False)
                    for x in string_values[0]
                ]
                result_is_not_nan = [
                    not math.isnan(x) if not isinstance(x, str) else x for x in results
                ]
                x_pos = np.array(sizes_to_compare)[result_is_not_nan]
                results = np.array(
                    [
                        get_number_from_val_plusminus_error(x, get_error=False)
                        for x in string_values[0]
                    ]
                )[result_is_not_nan]
                errors = np.array(
                    [
                        get_number_from_val_plusminus_error(x, get_error=True)
                        for x in string_values[0]
                    ]
                )[result_is_not_nan]
                ax.errorbar(
                    x_pos,
                    results,
                    yerr=errors,
                    alpha=0.7,
                    label=model_name,
                    capsize=0,
                    fmt="o",
                    ms=0,
                    linestyle="-",
                    elinewidth=0,
                )
            ax.plot(
                [x_pos.min() - 2, x_pos.max() + 2],
                [baseline, baseline],
                linestyle="dotted",
                color="k",
                label="fraction positive examples",
            )
            ax.legend()
            ax.set_ylabel("Average Precision")
            ax.set_xlabel("Training samples")
            ax.set_xticks(sizes_to_compare)
            ax.set_xlim([x_pos.min() - 2, x_pos.max() + 2])
            ax.set_xticklabels(sizes_to_compare)
            ax.set_title(f"Performance on task {task}")
            if results_dir is not None:
                plt.savefig(os.path.join(results_dir, f"{task}.png"))
            plt.show(fig)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to plot with exception: {e}")
            continue


def get_run_csvs_from_mountpath(mount_path: str, run: Run) -> List[str]:
    # designed to be used when aml experiment storage is mounted.
    if os.path.isdir(mount_path):
        print(f"Using mount path {mount_path}.")
        # address files using aml syntax
        run_mnt_dir = os.path.join(mount_path, f"dcid.{run._run_id}")
        try:
            csv_files = glob(os.path.join(run_mnt_dir, f"outputs/*.csv"))
        except Exception as e:
            print(f"Run {run._run_id} directory not found in mount {mount_path}." f" {e}")
        return csv_files

    raise ValueError(f"{mount_path} is not a directory, cannot read CSVs from here")


def load_model_results(
    file: str, model_label: str, TRAIN_SIZES_TO_COMPARE: List[int] = [16, 32, 64, 128, 256]
) -> pd.DataFrame:
    df = pd.read_csv(file)

    # keep only columns we need (statistics + results at requested train sizes)
    columns_to_keep = ["TASK_ID", "fraction_positive_train", "fraction_positive_test"]
    if "EC_super_class" in df.columns:
        columns_to_keep.append("EC_super_class")
    columns_to_keep.extend(f"{train_size}_train" for train_size in TRAIN_SIZES_TO_COMPARE)
    todrop = list(set(df.columns) - set(columns_to_keep))
    df.drop(columns=todrop, inplace=True)

    df = df.astype({"TASK_ID": str})

    # rename data columns so that we can do an outer merge later
    df.rename(
        columns={
            f"{train_size}_train": f"{train_size}_train ({model_label})"
            for train_size in TRAIN_SIZES_TO_COMPARE
        },
        inplace=True,
    )

    # Strip the CHEMBL prefix for readability:
    df["TASK_ID"] = df.apply(
        lambda row: row["TASK_ID"][len("CHEMBL") :]
        if row["TASK_ID"].startswith("CHEMBL")
        else row["TASK_ID"],
        axis=1,
    )

    return df


def merge_loaded_dfs(
    df_list: List[pd.DataFrame],
) -> pd.DataFrame:
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = merged_df.merge(
            df, how="outer", on=["TASK_ID", "fraction_positive_train", "fraction_positive_test"]
        )

    # Average out the columns on fractions of positives, which may have minor differences between runs due
    # to small differences in (stratified) sampling in some corner cases:
    frac_cols = ["fraction_positive_train", "fraction_positive_test"]
    merged_df_without_fracs = (
        merged_df.drop(columns=frac_cols)
        .groupby("TASK_ID")
        .agg(lambda x: np.nan if x.isnull().all() else x.dropna())
        .reset_index()
    )
    merge_on = ["TASK_ID"]
    if "EC_super_class" in merged_df.columns:
        merge_on.append("EC_super_class")
        merged_df = (
            merged_df.groupby(["TASK_ID"])
            .mean()
            .reset_index()
            .merge(merged_df_without_fracs, on=merge_on)
            # .astype({"EC_super_class": int})
        )
    else:
        merged_df = (
            merged_df.groupby(["TASK_ID"])
            .mean()
            .reset_index()
            .merge(merged_df_without_fracs, on=merge_on)
        )

    return merged_df


def load_data(
    model_summaries: Dict[str, str], TRAIN_SIZES_TO_COMPARE: List[int] = [16, 32, 64, 128, 256]
) -> pd.DataFrame:

    # load all the data in to a list
    df_list = []
    for model_name, model_summary_path in model_summaries.items():
        print(f"Loading data for {model_name} from {model_summary_path}.")
        df_list.append(load_model_results(model_summary_path, model_name, TRAIN_SIZES_TO_COMPARE))

    merged_df = merge_loaded_dfs(df_list)

    return merged_df


def expand_values(
    df: pd.DataFrame,
    model_summaries: Dict[str, str],
    TRAIN_SIZES_TO_COMPARE: List[int] = [16, 32, 64, 128, 256],
) -> pd.DataFrame:

    all_df = df.copy()
    for num_samples in [16, 32, 64, 128, 256]:
        for model_name in model_summaries.keys():
            all_df[f"{num_samples}_train ({model_name}) std"] = all_df.apply(
                lambda row: get_number_from_val_plusminus_error(
                    row[f"{num_samples}_train ({model_name})"], get_error=True
                ),
                axis=1,
            )
            all_df[f"{num_samples}_train ({model_name}) val"] = all_df.apply(
                lambda row: get_number_from_val_plusminus_error(
                    row[f"{num_samples}_train ({model_name})"], get_error=False
                ),
                axis=1,
            )

    return all_df


def plot_task_performances_byid(
    merged_df: pd.DataFrame,
    model_summaries: Dict[str, str],
    num_samples_choice: int,
    plot_output_dir: str = None,
    highlight_class: Optional[int] = None,
) -> None:

    frac_positives = merged_df["fraction_positive_train"]
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax.flatten()

    frac_pos_to_auprc_ax = ax[0]
    assay_id_to_improv_ax = ax[1]

    # Draw baseline diagonal:
    n = np.linspace(0, 1.0, 100)
    frac_pos_to_auprc_ax.plot(n, n, color="black")

    for i, model_name in enumerate(model_summaries.keys()):
        color = plt.get_cmap("plasma").colors[i * 30]
        # Get AUPRC for each model, to plot against fraction of posirives
        model_auprcs = [
            get_number_from_val_plusminus_error(model_result)
            for model_result in merged_df[f"{num_samples_choice}_train ({model_name})"].values
        ]
        frac_pos_to_auprc_ax.scatter(
            frac_positives, model_auprcs, s=100, marker="+", label=model_name, color=color
        )

        # select section to highlight
        if highlight_class is not None:
            hdf = merged_df[merged_df.EC_super_class == highlight_class].index
            nhdf = merged_df[merged_df.EC_super_class != highlight_class].index
            highlight_class_str = f", {highlight_class}"
        else:
            hdf = merged_df.index
            nhdf = None
            highlight_class_str = ""

        # Compute improvement over weighted random coinflip:
        model_auprc_diff_to_random = (
            merged_df.apply(
                lambda row: get_number_from_val_plusminus_error(
                    row[f"{num_samples_choice}_train ({model_name})"]
                ),
                axis=1,
            ).fillna(0)
            - frac_positives
        )
        assay_id_to_improv_ax.scatter(
            merged_df.iloc[hdf]["TASK_ID"],
            model_auprc_diff_to_random.iloc[hdf],
            s=100,
            marker="+",
            label=f"{model_name}{highlight_class_str}",
            color=color,
        )
        if nhdf is not None:
            assay_id_to_improv_ax.scatter(
                merged_df.iloc[nhdf]["TASK_ID"],
                model_auprc_diff_to_random.iloc[nhdf],
                s=100,
                marker="+",
                label=f"{model_name}",
                color=color,
                alpha=0.3,
            )

    frac_pos_to_auprc_ax.set_xlabel("fraction positive points")
    frac_pos_to_auprc_ax.set_ylabel(f"Average precision with {num_samples_choice} train points")
    frac_pos_to_auprc_ax.legend()
    frac_pos_to_auprc_ax.set_xlim([0.29, 0.51])
    frac_pos_to_auprc_ax.set_title(f"Class imbalance vs. AUPRC: $|T_s|$ = {num_samples_choice}")

    assay_id_to_improv_ax.set_xlabel("TASK ID")
    assay_id_to_improv_ax.set_ylabel(f"AUPRC gain with {num_samples_choice} train points")
    assay_id_to_improv_ax.set_ylim([-0.01, 0.42])
    assay_id_to_improv_ax.legend()
    assay_id_to_improv_ax.set_title(
        f"AUPRC improvement over random classification: $|T_s|$ = {num_samples_choice}"
    )

    assay_id_to_improv_ax.set_xticklabels(merged_df["TASK_ID"][0:-1:4], Rotation=90)

    assay_id_to_improv_ax.set_xticks(merged_df["TASK_ID"][0:-1:4])

    if plot_output_dir is not None:
        plt.savefig(
            os.path.join(
                plot_output_dir, f"model_comparison_{num_samples_choice}{highlight_class_str}.png"
            )
        )


def aggregate_by_class(
    ecmerged: pd.DataFrame,
    model_summaries: Dict[str, str],
    classes: List[int] = [1, 2, 3, 4, 5, 6, 7],
    num_samples: int = 16,
) -> pd.DataFrame:

    frac_positives = ecmerged["fraction_positive_train"]
    aggregation = None
    for highlight_class in classes:

        hdf = ecmerged[ecmerged.EC_super_class == highlight_class].index
        aggresults = pd.DataFrame()  # TODO: store results for each class
        aggerrors = pd.DataFrame()

        for model_name in model_summaries.keys():

            aggresults[f"{num_samples}_train ({model_name})"] = (
                ecmerged.iloc[hdf][f"{num_samples}_train ({model_name}) val"]
                - frac_positives.iloc[hdf]
            )
            # store to allow error propagation if only one measurement in category
            aggerrors[f"{num_samples}_train ({model_name}) std"] = ecmerged.iloc[hdf][
                f"{num_samples}_train ({model_name}) std"
            ]

        # take the mean value for each category
        s = pd.Series(data=aggresults.mean().values, index=aggresults.mean().index)

        # get the errors in different cases
        if len(aggresults) <= 1:  # or aggerrors.notna().sum() <= 1:
            # case of only one task for this category just take the real error of that measurement
            e = pd.Series(
                data=aggerrors.transpose().iloc[:, 0].values,
                index=aggerrors.transpose().iloc[:, 0].index,
            )

        elif any(aggresults.notna().sum()) <= 1:
            # only one task successful measurement for some models (not all)
            # for the models with only one successful measurement we use the measurement error
            empty_errors = pd.Series()
            for model_name in model_summaries.keys():
                if aggresults[f"{num_samples}_train ({model_name})"].notna().sum() == 1:
                    e = pd.Series(
                        data=aggerrors[f"{num_samples}_train ({model_name}) std"].dropna().values,
                        index=[f"{num_samples}_train ({model_name}) std"],
                    )
                elif aggresults[f"{num_samples}_train ({model_name})"].notna().sum() == 0:
                    e = pd.Series(
                        data=float("NaN"), index=[f"{num_samples}_train ({model_name}) std"]
                    )
                else:
                    e = pd.Series(
                        data=(
                            aggresults[f"{num_samples}_train ({model_name})"].std()
                            / np.sqrt(
                                aggresults[f"{num_samples}_train ({model_name})"].notna().sum()
                            )
                        ),
                        index=[f"{num_samples}_train ({model_name}) std"],
                    )
                if len(empty_errors) == 0:

                    empty_errors = e

                else:
                    empty_errors = pd.concat([empty_errors, e], axis=0)

            e = empty_errors
        else:
            # multiple measurements for category, take std error in mean
            e = pd.Series(
                data=(aggresults.std() / np.sqrt(aggresults.notna().sum())).values,
                index=aggerrors.transpose().iloc[:, 0].index,
            )

        aggs = pd.concat([s, e])
        total_df = (
            pd.DataFrame(aggs, columns=[str(highlight_class)])
            .transpose()
            .reset_index()
            .rename(columns={"index": "EC_category"})
        )

        if aggregation is None:
            aggregation = total_df
        else:
            aggregation = pd.concat([aggregation, total_df], axis=0)

    # aggregate across all EC categories
    hdf = ecmerged.index
    aggresults = pd.DataFrame()
    aggerrors = pd.DataFrame()
    highlight_class = "all"
    for model_name in model_summaries.keys():

        aggresults[f"{num_samples}_train ({model_name})"] = (
            ecmerged[f"{num_samples}_train ({model_name}) val"] - frac_positives.iloc[hdf]
        )
        aggerrors[f"{num_samples}_train ({model_name}) std"] = ecmerged.iloc[hdf][
            f"{num_samples}_train ({model_name}) std"
        ]

    s = pd.Series(data=aggresults.mean().values, index=aggresults.mean().index)
    e = pd.Series(
        data=(aggresults.std() / np.sqrt(aggresults.notna().sum())).values,
        index=aggerrors.transpose().iloc[:, 0].index,
    )

    aggs = pd.concat([s, e])
    total_df = (
        pd.DataFrame(aggs, columns=[str(highlight_class)])
        .transpose()
        .reset_index()
        .rename(columns={"index": "EC_category"})
    )
    aggregation = pd.concat([aggregation, total_df], axis=0)

    return aggregation


def calculate_delta_auprc(
    df: pd.DataFrame,
    model_summaries: Dict[str, str],
    train_samples_to_compare: List[int] = [16, 32, 64, 128, 256],
) -> pd.DataFrame:

    extend_df = df.copy()
    for num_samples in train_samples_to_compare:
        for model_name in model_summaries.keys():
            extend_df[f"{num_samples}_train ({model_name}) val delta-auprc"] = extend_df.apply(
                lambda row: row[f"{num_samples}_train ({model_name}) val"]
                - row["fraction_positive_train"],
                axis=1,
            )

    return extend_df
