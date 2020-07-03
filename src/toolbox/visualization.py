from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from matplotlib.gridspec import GridSpec

sns.set_style("white")

Query = Tuple[str, str]
Location = Tuple[float, float, float]  # start, end, length
Instance = Tuple[Query, Location]
Rating = List[float]
Metrics = Dict[str, float]
Prediction = Tuple[Query, List[Location], Rating]
Result = Tuple[Query, List[Location], Rating, Metrics]


def plot_performance_per_class(
    metrics_per_cls: Dict[str, List[Result]]
) -> plt.Figure:
    order = np.argsort(
        [metrics["rate_success"] for metrics in metrics_per_cls.values()]
    )[::-1]
    keys = list(metrics_per_cls.keys())
    n_success = list(
        [metrics["n_success"] for metrics in metrics_per_cls.values()]
    )
    n_instance = list(
        [metrics["n_instance"] for metrics in metrics_per_cls.values()]
    )

    keys = [keys[i] for i in order]
    n_success = [n_success[i] for i in order]
    n_instance = [n_instance[i] for i in order]

    N_cls = len(keys)

    fig = plt.figure(figsize=(15, 3))
    _ = plt.xticks(np.arange(N_cls), keys, rotation=60, fontsize=14)

    plt.bar(np.arange(N_cls), n_instance, color="#606060", label="failure")
    plt.bar(np.arange(N_cls), n_success, color="#9c47ff", label="success")
    plt.legend(frameon=False, fontsize=14)
    sns.despine()

    return fig


def plot_ranking_comparison(
    metrics_per_cls_a: Dict[str, dict],
    metrics_per_cls_b: Dict[str, dict],
    label_a: str,
    label_b: str,
):

    fig = plt.figure(figsize=(10, 10))
    rate_success_a = [
        metrics["rate_success"] for metrics in metrics_per_cls_a.values()
    ]
    n_success_a = [
        metrics["n_success"] for metrics in metrics_per_cls_a.values()
    ]
    n_instance_a = [
        metrics["n_instance"] for metrics in metrics_per_cls_a.values()
    ]

    rate_success_b = [
        metrics["rate_success"] for metrics in metrics_per_cls_b.values()
    ]
    n_success_b = [
        metrics["n_success"] for metrics in metrics_per_cls_b.values()
    ]
    n_instance_b = [
        metrics["n_instance"] for metrics in metrics_per_cls_b.values()
    ]

    gs = GridSpec(1, 4)
    axes_a = fig.add_subplot(gs[0, 0])
    axes_a.barh(
        rate_success_a, n_instance_a, height=0.01, color="darkgrey", alpha=0.7
    )
    axes_a.barh(rate_success_a, n_success_a, height=0.01, color="limegreen")
    axes_a.invert_xaxis()
    axes_a.set_ylim(0, 1)

    axes_b = fig.add_subplot(gs[0, 3])
    axes_b.barh(
        rate_success_b, n_instance_b, height=0.01, color="darkgrey", alpha=0.7
    )
    axes_b.barh(rate_success_b, n_success_b, height=0.01, color="limegreen")
    axes_b.set_yticks([])
    axes_b.set_ylim(0, 1)

    axes_c = fig.add_subplot(gs[0, 1:3])
    axes_c.set_ylim(0, 1)
    axes_c.set_xlim(0, 1)
    xmin, xmax = axes_c.get_xlim()
    for cls_label in metrics_per_cls_a.keys():
        rate_a = metrics_per_cls_a[cls_label]["rate_success"]
        rate_b = metrics_per_cls_b[cls_label]["rate_success"]

        axes_c.plot((0.0, 1), (rate_a, rate_b), color="darkgrey", linewidth=1)
        text_a = axes_c.text(
            xmin,
            rate_a,
            cls_label,
            verticalalignment="center",
            fontsize=16,
            bbox=dict(facecolor="w"),
        )
        text_b = axes_c.text(
            xmax,
            rate_b,
            cls_label,
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=16,
            bbox=dict(facecolor="w"),
        )
    axes_c.axis("off")


def plot_performance_per_duration(
    results: List[Result], groundtruth: List[Instance], ax=None
) -> None:

    # fig = plt.figure()
    failure = []
    success = []
    for r, gt_dat in zip(results, groundtruth):
        _, (s, e, l) = gt_dat
        duration = (e - s) / l
        is_succeed = r[-1]["R@1 IoU>0.5"]
        if is_succeed:
            success.append(duration)
        else:
            failure.append(duration)

    ax.hist(
        [success, failure],
        bins=20,
        stacked=True,
        label=["success", "failure"],
        color=["#9c47ff", "#606060"],
    )
    ax.legend(frameon=False, fontsize=14)
