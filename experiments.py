# from albumentations.random_utils import normal

from scipy.stats import ks_2samp
from seaborn import FacetGrid
import warnings

warnings.filterwarnings("ignore")
from experiments.dataset_analysis import *
from experiments.runtime_classification import *
np.set_printoptions(precision=3, suppress=True)


def run_rv_experiments():
    """
          Runtime Verification
      """
    if not os.path.exists("ood_detector_data"):
        os.makedirs("ood_detector_data")

    for batch_size in BATCH_SIZES:
        print(f"Running batch size {batch_size}")
        ood_detector_correctness_prediction_accuracy(batch_size, shift="")

    ood_accuracy_vs_pred_accuacy_plot(1)
    # single batch size

    # simple batching
    ood_verdict_plots_batched()

    plot_batching_effect("NICO", "entropy")

    for batch_size in BATCH_SIZES[1:-1]:
        debiased_ood_detector_correctness_prediction_accuracy(batch_size)

    ood_verdict_plots_batched()


if __name__ == '__main__':
    #accuracies on each dataset
    run_rv_experiments()
    # run_loss_regression_experiments()
    # run_pra_experiments()
    # run_appendix_experiments()


