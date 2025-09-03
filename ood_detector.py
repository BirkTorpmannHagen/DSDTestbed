import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp

def l2_distance(a, b):
    """
    Compute the L2 distance between two vectors.
    :param a: First vector (numpy array)
    :param b: Second vector (numpy array)
    :return: L2 distance (float)
    """
    return np.linalg.norm(a - b)

def ks_distance(a, b):
    try:
        return ks_2samp(a[a!=0], b[b!=0]).pvalue
    except ValueError:
        return -1




def get_optimal_threshold(ind, ood):
    merged = np.concatenate([ind, ood])
    max_acc = 0
    threshold = 0
    if ind.mean()<ood.mean():
        higher_is_ood = True
    else:
        higher_is_ood = False

    #if linearly seperable, set the threshold to the middle
    if ind.max()<ood.min() and higher_is_ood:
        return (ind.max()+ood.min())/2
    if ood.max()<ind.min() and not higher_is_ood:
        return (ind.max() + ood.min()) / 2

    for t in np.linspace(merged.min(), merged.max(), 100):
        if higher_is_ood:
            ind_acc = (ind<t).mean()
            ood_acc = (ood>t).mean()
        else:
            ind_acc = (ind>t).mean()
            ood_acc = (ood<t).mean()
        bal_acc = 0.5*(ind_acc+ood_acc)
        if not higher_is_ood:
            if bal_acc>=max_acc: #ensures that the threshold is near ind data for highly seperable datasets
                max_acc = bal_acc
                threshold = t
        else:
            if bal_acc>max_acc:
                max_acc = bal_acc
                threshold = t


    return threshold


class OODDetector:
    def __init__(self, df, ood_val_shift, threshold_method="val_optimal"):
        assert df["feature_name"].nunique() == 1
        assert df["Dataset"].nunique() == 1
        if "k" in df.columns:
            assert df["k"].nunique() == 1, "OODDetector only works with k=1 due to the thresholding scheme"
        self.df = df
        self.threshold_method = threshold_method

        # self.ind_val = df[(df["shift"] == "ind_val")&(~df["ood"])]
        # self.ood_val = df[(df["shift"] == ood_val_shift)&(df["ood"])]
        self.ind_val = df[~df["ood"]]
        self.ood_val = df[df["ood"]]
        self.higher_is_ood = self.ood_val["feature"].mean() >self.ind_val["feature"].mean()
        if threshold_method == "val_optimal":
            self.threshold = get_optimal_threshold(self.ind_val["feature"], self.ood_val["feature"])
        if threshold_method == "ind_span":
            lower = self.ind_val["feature"].quantile(0.01)
            upper = self.ind_val["feature"].quantile(0.99)
            self.threshold = [lower,upper]
        if threshold_method=="logistic":
            self.logreg = LogisticRegression()
            min_len = min(len(self.ind_val), len(self.ood_val))

            #avoid poor balancing
            if len(self.ind_val) > min_len:
                self.ind_val = self.ind_val.sample(min_len, random_state=42)
            if len(self.ood_val) > min_len:
                self.ood_val = self.ood_val.sample(min_len, random_state=42)

            features = np.concatenate([self.ind_val["feature"].values, self.ood_val["feature"].values]).reshape(-1, 1)
            labels = np.concatenate([self.ind_val["ood"].astype(int).values, self.ood_val["ood"].astype(int).values])
            self.logreg.fit(features, labels)





    def kde(self):
        sns.kdeplot(self.ind_val["feature"], label="ind_val", color="blue")
        sns.kdeplot(self.ood_val["feature"], label="ood_val", color="red")
        if self.threshold_method == "ind_span":
            plt.axvline(self.threshold[0], color="red", linestyle="--")
            plt.axvline(self.threshold[1], color="red", linestyle="--")
        elif self.threshold_method == "val_optimal":
            plt.axvline(self.threshold, color="red", linestyle="--")
        plt.show()

    def predict(self, batch):
        # if isinstance(batch["feature"], pd.Series): #if it is a batch
        #     feature = batch["feature"].mean()
        # else:
        #     feature = batch["feature"]
        feature = batch["feature"]
        if self.threshold_method == "ind_span":
            return not (self.threshold[0] <= feature <= self.threshold[1])
        elif self.threshold_method=="val_optimal":
            if self.higher_is_ood:
                return  feature > self.threshold
            else:
                return feature < self.threshold
        elif self.threshold_method == "logistic":
            if feature==np.inf:
                print(batch)
            output = bool(self.logreg.predict(np.array(feature).reshape(1, -1))[0])
            return output
        else:
            return NotImplementedError



    def get_tpr(self, data):
        subset = data[data["ood"]]
        if subset.empty:
            return 1
        tpr =  subset.apply(lambda row: self.predict(row), axis=1).mean()
        return tpr


    def get_tnr(self, data):
        subset = data[~data["ood"]]
        if subset.empty:
            return 1
        tnr =  1-subset.apply(lambda row: self.predict(row), axis=1).mean()
        return tnr

    def get_accuracy(self, data):
        return 0.5*(self.get_tpr(data)+self.get_tnr(data))

    def get_metrics(self, data):
        return self.get_tpr(data), self.get_tnr(data), self.get_accuracy(data)


    def plot_hist(self):
        sns.histplot(self.df, x="feature", hue="ood", alpha=0.5)
        plt.title(f"{self.df['feature_name'].unique()[0]} - {self.df['Dataset'].unique()[0]}")

        if self.threshold_method == "ind_span":
            plt.axvline(self.threshold[0], color="red", linestyle="--")
            plt.axvline(self.threshold[1], color="red", linestyle="--")
        elif self.threshold_method == "val_optimal":
            plt.axvline(self.threshold, color="red", linestyle="--")

        # plt.title(self.get_accuracy(self.df))
        plt.show()

    def get_likelihood(self):
        return self.get_tpr(self.df), self.get_tpr(self.df)




