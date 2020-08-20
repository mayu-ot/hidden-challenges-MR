from typing import Dict
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.toolbox.utils import sentence2token, _load_top_actions
from src.toolbox.eval import evaluate, accumulate_metrics
from src.toolbox.data_converters import (
    CharadesSTA2Instances,
    ActivityNetCap2Instances,
)
from src.toolbox.baseline import SegmentGeneratorKDE, predict
import seaborn as sns
import click

sns.set_style("white")


def load_activitynet_dataset(split: str):
    if split == "test":
        split = "val_2"

    dataset = ActivityNetCap2Instances(
        json.load(open(f"data/raw/activitynet/{split}.json"))
    )

    return dataset


def load_charades_dataset(split: str):
    dataset = CharadesSTA2Instances(
        pd.read_csv(f"data/processed/charades/charades_{split}.csv")
    )
    return dataset


def load_dataset(split: str, dataname: str):
    if dataname == "charades":
        dataset = load_charades_dataset(split)
    elif dataname == "activitynet":
        dataset = load_activitynet_dataset(split)
    return dataset


def train_prior_only(train_data):
    model = SegmentGeneratorKDE()
    model.fit("base", train_data)
    return model


def train_action_aware_blind(train_data, dataname: str):
    sentences = [query[1] for query, _ in train_data]
    tokens = [sentence2token(x) for x in sentences]

    top_actions = _load_top_actions(dataname)
    model = SegmentGeneratorKDE()
    for action in top_actions:
        indices = [i for i, query in enumerate(tokens) if action in query[0]]
        sub_train = [train_data[i] for i in indices]
        model.fit(action, sub_train)
    model.fit("base", train_data)
    return model


def train(dataname: str):
    train_data = load_dataset("train", dataname)

    prior_only_model = train_prior_only(train_data)
    action_aware_model = train_action_aware_blind(train_data, dataname)

    return prior_only_model, action_aware_model


def display_score(bar, color="w"):
    plt.text(
        bar.get_x() + bar.get_width() * 0.5,
        bar.get_height() - 6,
        f"{bar.get_height():.1f}",
        horizontalalignment="center",
        fontsize=12,
        color=color,
    )


def plot_performance_summary(
    summary: Dict[str, float], title: str, outfile: str
):
    plt.figure(figsize=(6, 4))
    c = ["#d602ee", "#df55f2", "#e98df5"]
    x = np.arange(3) + 0.2

    for metric in ["R@1", "R@5", "R@10"]:
        vals = [
            summary[k] * 100 for k in summary.keys() if k.split()[0] == metric
        ]
        bars = plt.bar(x, vals, width=0.3, color=c.pop(0), label=metric)
        for b in bars:
            display_score(b, color="w")
        x += 0.3

    plt.xticks(
        ticks=np.arange(3) + 0.4,
        labels=["IoU>0.3", "IoU>0.5", "IoU>0.7"],
        ha="center",
    )

    plt.legend()
    plt.title(title)
    sns.despine(left=False)

    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    plt.savefig(outfile, bbox_inches="tight")


def eval_model(model, dataname: str):
    test_data = load_dataset("test", dataname)

    predictions = predict(model, test_data, 0.45)
    results = evaluate(test_data, predictions)
    summary = accumulate_metrics(results)

    return predictions, results, summary


@click.command()
@click.argument("dataname", type=str)
def main(dataname):
    prior_only_model, action_aware_model = train(dataname)

    print("prior-only baseline:")
    po_pred, po_results, po_summary = eval_model(prior_only_model, dataname)

    print("action-aware blind baseline:")
    act_pred, act_results, act_summary = eval_model(
        action_aware_model, dataname
    )

    plot_performance_summary(
        po_summary,
        title="Prior-Only",
        outfile=f"reports/figures/performance-analysis/{dataname}/summary-prior-only.pdf",
    )

    plot_performance_summary(
        act_summary,
        title="Action-Aware Blind",
        outfile=f"reports/figures/performance-analysis/{dataname}/summary-action-aware.pdf",
    )


if __name__ == "__main__":
    main()
