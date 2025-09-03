import itertools
import os.path

import numpy as np
import pandas
import pandas as pd
import seaborn
from matplotlib import pyplot
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count
from matplotlib.patches import Patch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import seaborn as sns
import matplotlib.pyplot as plt
from ood_detector import OODDetector
from itertools import product
from utils import *


def parallel_compute_ood_detector_prediction_accuracy(data_filtered, threshold_method, dataset, feature, ood_perf, perf_calibrated):
    data_dict = []
    for ind_val_fold in ["ind_val", "ind_test"]:
        for ood_val_fold in data_filtered["shift"].unique():
            data_copy = data_filtered.copy()
            if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold in SYNTHETIC_SHIFTS:
                # dont calibrate on ind data or synthetic ood data
                continue
            data_train = data_copy[
                (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == ind_val_fold)]
            dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)
            # dsd.kde()
            for ood_test_fold in data_filtered["fold"].unique():

                if ood_test_fold in ["train", "ind_val", "ind_test"]:
                    continue
                if perf_calibrated:
                    data_copy["ood"] = ~data_copy["correct_prediction"]
                ind_test_fold = "ind_test" if ind_val_fold == "ind_val" else "ind_val"
                data_test = data_copy[(data_copy["fold"] == ood_test_fold) | (data_copy["fold"] == ind_test_fold)]
                shift = ood_test_fold.split("_")[0]
                shift_intensity = ood_test_fold.split("_")[-1] if "_" in ood_test_fold else "Organic"
                if ood_perf and not perf_calibrated:
                    data_copy["ood"] = ~data_copy["correct_prediction"]
                tpr, tnr, ba = dsd.get_metrics(data_test)
                if np.isnan(ba):
                    continue
                data_dict.append({"Dataset": dataset, "feature_name": feature, "Threshold Method": threshold_method,
                     "OoD==f(x)=y": ood_perf, "Performance Calibrated": perf_calibrated,
                     "OoD Val Fold": ood_val_fold, "InD Val Fold": ind_val_fold,
                     "OoD Test Fold": ood_test_fold, "InD Test Fold": ind_test_fold, "Shift": shift,
                     "Shift Intensity": shift_intensity, "tpr": tpr, "tnr": tnr, "ba": ba})
    return data_dict


def _compute_one(args):
    (
        data_filtered, threshold_method, dataset, feature,
        ood_perf, perf_calibrated
    ) = args
    try:
        out = parallel_compute_ood_detector_prediction_accuracy(
            data_filtered, threshold_method, dataset, feature,
            ood_perf, perf_calibrated
        )
        return out  # may be None if function early-continues
    except Exception as e:
        # optional: return diagnostics instead of raising to avoid killing the pool
        return {
            "Dataset": dataset,
            "feature_name": feature,
            "Threshold Method": threshold_method,
            "OoD==f(x)=y": ood_perf,
            "Performance Calibrated": perf_calibrated,
            "error": str(e),
        }

def _make_jobs_for_feature(data_filtered, dataset, feature):
    # (ood_perf, perf_calibrated) pairs except the skipped case
    pairs = [(o, p) for o, p in product([True, False], [True, False]) if not (p and not o)]
    # cartesian product over threshold methods
    jobs = [
        (data_filtered, tm, dataset, feature, o, p)
        for (o, p) in pairs
        for tm in THRESHOLD_METHODS
    ]
    return jobs

def ood_detector_correctness_prediction_accuracy(batch_size, shift="normal"):
    """
    Evaluates OOD detector correctness prediction accuracy for all datasets and features and saves to CSV files.
    """
    df = load_all(prefix="fine_data", batch_size=batch_size, shift=shift, samples=100)
    df = df[df["fold"] != "train"]

    # Precompute total jobs for progress bar
    total_jobs = 0
    for dataset in DATASETS:
        data_dataset = df[df["Dataset"] == dataset]
        for feature in DSDS:
            data_filtered = data_dataset[data_dataset["feature_name"] == feature]
            if data_filtered.empty:
                continue
            total_jobs += len(_make_jobs_for_feature(data_filtered, dataset, feature))

    results_all = []
    # choose pool size
    n_procs = max(1, cpu_count() - 1)

    with Pool(processes=n_procs) as pool, tqdm(total=total_jobs, desc="Computing") as pbar:
        for dataset in DATASETS:
            if dataset=="Polyp":
                continue
            data_dataset = df[df["Dataset"] == dataset]
            for feature in DSDS:
                data_filtered = data_dataset[data_dataset["feature_name"] == feature]
                if data_filtered.empty:
                    print(f"No data for {dataset} {feature}")
                    continue

                jobs = _make_jobs_for_feature(data_filtered, dataset, feature)

                # stream results as they complete; update pbar per completed task
                for out in pool.imap_unordered(_compute_one, jobs, chunksize=1):
                    pbar.update(1)
                    if out is None:
                        continue
                    results_all.append(out)

    if not results_all:
        print("No results produced.")
        return
    flat_success = [row for out in results_all if isinstance(out, list) for row in out]
    flat_errors = [out for out in results_all if isinstance(out, dict) and "error" in out]
    data = pd.DataFrame(flat_success)
    print(data)
    if not data.empty:
        data["feature_name"].replace(DSD_PRINT_LUT, inplace=True)

    if flat_errors:
        pd.DataFrame(flat_errors).to_csv(f"ood_detector_data/ood_detector_errors_{batch_size}.csv", index=False)

    for dataset in DATASETS:
        data_ds = data[data["Dataset"] == dataset]
        if data_ds.empty:
            continue
        data_ds.to_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)

def get_all_ood_detector_data(batch_size, filter_thresholding_method=False, filter_ood_correctness=False, filter_correctness_calibration=False, filter_organic=False, filter_best=False):
    dfs = []
    for dataset, feature in itertools.product(DATASETS, DSDS):
        dfs.append(pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv"))
    df = pd.concat(dfs)
    if filter_thresholding_method:
        df = df[df["Threshold Method"] == "val_optimal"]
    if filter_ood_correctness:
        df = df[df["OoD==f(x)=y"] == False]
    if filter_correctness_calibration:
        df = df[df["Performance Calibrated"] == False]
    if filter_organic:
        df = df[~(df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS) )&(~df["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))]
    if filter_best:
        meaned_ba = df.groupby(["Dataset", "feature_name"])["ba"].mean().reset_index()
        best_ba = meaned_ba.loc[meaned_ba.groupby("Dataset")["ba"].idxmax()]
        df = df.merge(best_ba[["Dataset", "feature_name"]], on=["Dataset", "feature_name"], how="inner")



    return df

def ood_rv_accuracy_by_thresh_and_stuff(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True)
    print(df.groupby(["Threshold Method", "OoD==f(x)=y", "Performance Calibrated"])[["ba"]].agg(["min", "mean", "max"]))

def ood_rv_accuracy_by_dataset_and_feature(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True, filter_thresholding_method=True, filter_correctness_calibration=True)
    print(df.groupby(["OoD==f(x)=y", "Dataset", "feature_name"])["ba"].mean())


def ood_accuracy_vs_pred_accuacy_plot(batch_size):
    """
    Plot the relationship between OoD detection rate and generalization gap
    """
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=False,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=True)
    df = df[df["OoD==f(x)=y"] == True]  # only OOD performance


    # df = df[~((df["OoD==f(x)=y"] == True)&(~df["Performance Calibrated"]))]  # only OOD performance
    #get only the shifts that affect the performance of the OOD detector
    df_raw = load_all(batch_size, shift="", prefix="fine_data")

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "fold"])["correct_prediction"].mean().reset_index()

    ood_accs = df.groupby(["Dataset", "OoD Test Fold", "OoD==f(x)=y"])["tpr"].mean().reset_index()
    ind_accs = df.groupby(["Dataset", "InD Test Fold", "OoD==f(x)=y"])["tnr"].mean().reset_index()
    ind_accs["tnr"]=1-ind_accs["tnr"]
    ind_accs.rename(columns={"InD Test Fold":"fold", "tnr":"Detection Rate"}, inplace=True)
    ood_accs.rename(columns={"OoD Test Fold":"fold", "tpr":"Detection Rate"}, inplace=True)

    merged = pd.concat([ood_accs, ind_accs], ignore_index=True)
    merged = merged.merge(acc_by_dataset_and_shift, on=["Dataset", "fold"])
    merged["Shift"] = merged["fold"].apply(lambda x: x.split("_")[0] if "_" in x else "Organic")
    merged["Organic"] = merged["Shift"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")

    acc = merged.groupby(["Dataset", "fold"], as_index=False)["correct_prediction"].mean()

    # pull the per-dataset ind_val baseline
    ind = (acc.loc[acc["fold"] == "ind_val", ["Dataset", "correct_prediction"]]
           .rename(columns={"correct_prediction": "ind_val_acc"}))

    # join baseline back to every shift of the same dataset
    acc = acc.merge(ind, on="Dataset", how="left")

    # absolute and relative differences vs ind_val

    acc["Generalization Gap"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Accuracy"] = acc["correct_prediction"]
    # acc["Generalization Gap"] = acc["acc_diff"] / acc["ind_val_acc"]  # e.g., 0.10 == +10%
    # acc["Generalization Gap"] = - acc["Generalization Gap"] * 100  # convert to percentage
    merged = merged.merge(acc, on=["Dataset", "fold"], how="left")
    print(merged)
    merged["shift"] = merged.replace(SHIFT_PRINT_LUT, inplace=True)

    hue_order = sorted(merged["Shift"].unique().tolist())
    print(hue_order)
    def plot_ideal_line(data, color=None, **kwargs):
        # Plot a diagonal line from (0, 0) to (1, 1)
        dataset = data["Dataset"].unique()[0]
        plt.axhline(1-DATASETWISE_RANDOM_CORRECTNESS[dataset], color="blue", linestyle="--", label="Maximum Detection Rate")

    g = sns.FacetGrid(merged, col="Dataset", sharex=False, sharey=False, col_wrap=3)
    # g.map_dataframe(sns.regplot, x="Generalization Gap", y="Detection Rate", robust=False, scatter=False)
    g.map_dataframe(sns.scatterplot, x="Generalization Gap", y="Detection Rate", hue="Shift", alpha=0.5, edgecolor=None, hue_order=hue_order)
    g.map_dataframe(plot_ideal_line)
    g.add_legend()
    g.set_axis_labels("Generalization Gap", "OoD Detection Rate")
    for ax in g.axes.flat:
        ax.set_ylim(0,1.1)
        # ax.set_xlim(0,1.1)
    plt.savefig("figures/tpr_v_acc.pdf")
    plt.show()


def debiased_ood_detector_correctness_prediction_accuracy(batch_size):
    df = load_all_biased(filter_batch=batch_size)
    df = df[df["fold"]!="train"]
    for dataset in DATASETS:
        data_dict = []
        data_dataset = df[df["Dataset"] == dataset]
        if dataset=="Polyp":
            print(data_dataset.head(10))
        with tqdm(total=df["feature_name"].nunique()*2 * 2, desc=f"Computing for {dataset}") as pbar:
            if os.path.exists(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv"):
                print("continuing...")
                continue
            for feature in DSDS:
                for k in [-1, 0, 1, 5]:

                    if feature == "knn" and k !=-1:
                        continue
                    if feature == "softmax" and dataset=="Polyp":
                        continue
                    if feature=="rabanser" and k==-1:
                        continue
                    data_filtered = data_dataset[(data_dataset["feature_name"]==feature)&(data_dataset["k"]==k)]
                    if data_filtered.empty:
                        print(f"empty for {dataset}, {feature}, {k})")
                        # input()
                        continue

                        # print("continuing")
                    for ood_perf in [True, False]:
                        for perf_calibrated in [True, False]:
                            if perf_calibrated and not ood_perf:
                                continue  # unimportant
                            for threshold_method in THRESHOLD_METHODS:
                                for ood_val_fold in data_filtered["shift"].unique():
                                    data_copy = data_filtered.copy()
                                    if ood_val_fold in ["train", "ind_val", "ind_test"]:
                                        continue
                                    data_train = data_copy[
                                        ((data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val") ) & (data_copy["bias"]=="RandomSampler")]
                                    if data_train.empty:
                                        print(f"No training data for {dataset} {feature}, {k}")
                                        continue
                                    dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)
                                    # dsd.kde()
                                    for ood_test_fold in data_filtered["shift"].unique():
                                        if ood_test_fold in ["train", "ind_val", "ind_test"]:
                                            continue
                                        for bias in SAMPLERS:
                                            if bias=="ClassOrderSampler" and dataset=="Polyp":
                                                continue

                                            if perf_calibrated:
                                                data_copy["ood"]=~data_copy["correct_prediction"]

                                            data_test = data_copy[((data_copy["shift"]==ood_test_fold)|(data_copy["shift"]=="ind_test"))&(data_copy["bias"]==bias)]

                                            if ood_perf and not perf_calibrated:
                                                data_copy["ood"]=~data_copy["correct_prediction"]
                                            tpr, tnr, ba = dsd.get_metrics(data_test)

                                            if np.isnan(ba):
                                                print("nan val!")
                                                continue
                                            data_dict.append(
                                                {"Dataset": dataset, "feature_name": feature, "Threshold Method": threshold_method,
                                                 "OoD==f(x)=y": ood_perf, "Performance Calibrated": perf_calibrated,
                                                 "OoD Val Fold": ood_val_fold, "OoD Test Fold":ood_test_fold, "bias":SAMPLER_LUT[bias], "k":k, "tpr": tpr, "tnr": tnr, "ba": ba}
                                            )
                                            pbar.set_description(f"Computing for {dataset}, {feature} {ood_perf} {ood_test_fold} {bias}; current ba: {ba}")
                            pbar.update(1)
            data = pd.DataFrame(data_dict)
            if not data.empty:
                data.replace(DSD_PRINT_LUT, inplace=True)
                data.to_csv(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)



def ood_verdict_plots_batched():
    dfs = []
    for dataset, batch_size in itertools.product(DATASETS, BATCH_SIZES):
        df = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
        df["batch_size"] = batch_size
        dfs.append(df)
    data = pd.concat(dfs)
    data = data[(~data["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))&(~data["OoD Val Fold"].isin(SYNTHETIC_SHIFTS))]

    data = data[(data["Threshold Method"]=="val_optimal")&(data["Performance Calibrated"]==False)]
    g = sns.FacetGrid(data, col="Dataset", row="OoD==f(x)=y", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="feature_name", markers=True, dashes=False)
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)

    g.add_legend(bbox_to_anchor=(0.7, 0.3), loc='center left', title="Feature", ncol=1)
    plt.savefig("batched_ood_verdict_accuracy.pdf")
    plt.show()

def plot_batching_effect(dataset, feature):
    df = load_pra_df(dataset, feature, batch_size=1)
    df_batched = load_pra_df(dataset, feature, batch_size=30)
    oods = df[(df["ood"])&(~df["correct_prediction"])]
    inds = df[(~df["ood"])&(df["correct_prediction"])]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
    plot_df = pd.concat([oods, inds])
    sns.kdeplot(plot_df, x="feature", hue="ood", fill=True, common_norm=False,ax=ax1)
    sns.kdeplot(df_batched, x="feature", hue="ood", fill=False, common_norm=False, ax=ax2, linestyle="--")
    plt.tight_layout()
    # plt.yscale("log")
    plt.xlim(0,2500)
    plt.savefig(f"{dataset}_{feature}_kdes.pdf")
    plt.show()

