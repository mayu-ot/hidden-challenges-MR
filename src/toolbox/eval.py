from typing import Tuple, List, Dict, Callable
import numpy as np
from .utils import sentence2token
from tqdm import tqdm

Query = Tuple[str, str]
Location = Tuple[float, float, float]  # start, end, length
Instance = Tuple[Query, Location]
Rating = List[float]
Prediction = Tuple[Query, List[Location], Rating]
Result = Tuple[Query, List[Location], Rating, dict]


def _tiou(pred: np.ndarray, gt: Tuple[float, float]):
    inter_left = np.maximum(pred[:, 0], gt[0])
    inter_right = np.minimum(pred[:, 1], gt[1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0], gt[0])
    union_right = np.maximum(pred[:, 1], gt[1])
    union = np.maximum(0.0, union_right - union_left)
    return 1.0 * inter / union


def evaluate(
    groundtruth: List[Instance],
    prediction: List[Prediction],
    top_k: List[int] = [1, 5, 10],
    iou_threshold: List[float] = [0.3, 0.5, 0.7],
) -> List[Result]:

    if len(groundtruth) != len(prediction):
        print(f"{len(groundtruth) - len(prediction)} missing instances")

    results = []

    for gt_instance in tqdm(groundtruth, desc="evaluating"):
        gt_query, gt_loc = gt_instance
        prediction_found = False

        for pred_instance in prediction:
            metrics = {}
            query, pred_locs, rating = pred_instance
            if gt_query == query:
                prediction_found = True
                if pred_locs[0][-1] != gt_loc[-1]:
                    raise RuntimeError("The video length does not match.")

                bbox = np.asarray(pred_locs)[:, :2]
                bbox = bbox[np.argsort(rating)[::-1]]
                overlap = _tiou(bbox, gt_loc[:2])

                for k in top_k:
                    for thresh in iou_threshold:
                        is_success = (overlap > thresh)[:k].any()
                        metrics[f"R@{k} IoU>{thresh:.1f}"] = is_success

                results.append((query, pred_locs, rating, metrics))
                break

        if not prediction_found:
            for k in top_k:
                for thresh in iou_threshold:
                    metrics[f"R@{k} IoU>{thresh:.1f}"] = False
            results.append((query, None, None, metrics))
            print(f"missing item: {gt_query}")

    return results


def accumulate_metrics(results: List[Result]) -> dict:
    metrics = results[0][-1]
    accum_metrics: dict = {metric_type: [] for metric_type in metrics.keys()}

    for _, _, _, metrics in results:

        for metric_type, is_success in metrics.items():
            accum_metrics[metric_type].append(is_success)

    summary = {
        metric_type: sum(all_result) / len(all_result)
        for metric_type, all_result in accum_metrics.items()
    }

    for metric_type, score in summary.items():
        print(f"{metric_type} {score:.2f}")

    return summary


def location_error(
    groundtruth: List[Instance],
    results: List[Result],
    metric_type: str = "R@1 IoU>0.5",
):
    errors = []

    for result in results:
        query = result[0]
        pred_locs = result[1]
        rating = result[2]
        metrics: dict = result[3]

        if metrics[metric_type]:
            continue

        for gt_query, gt_loc in groundtruth:
            if query == gt_query:
                top_pred = pred_locs[np.argmax(rating)]
                pred_start, pred_end, _ = top_pred
                gt_start, gt_end, length = gt_loc

                start_error = (pred_start - gt_start) / length
                end_error = (pred_end - gt_end) / length

                errors.append((query, (start_error, end_error)))
    return errors


def get_first_action(result: Result, action_vocab: List[str]) -> str:
    query, _, _, _ = result
    video_id, sentence = query
    actions, objects = sentence2token(sentence)

    if len(actions) == 0:
        action = "OoV"

    else:
        if actions[0] in action_vocab:
            action = actions[0]
        else:
            action = "OoV"

    return action


def categorize_results(
    results: List[Result], cat_fn=Callable
) -> Dict[str, List[Result]]:
    keyed_results: Dict[str, List[Result]] = {}
    for result in results:
        cls_label = cat_fn(result)
        keyed_results.setdefault(cls_label, []).append(result)
    return keyed_results


def summarize_results_per_class(keyed_results: dict) -> dict:
    metrics_per_cls = {}
    for cls_label, results in keyed_results.items():
        is_success = [metrics["R@1 IoU>0.5"] for _, _, _, metrics in results]
        n_success = sum(is_success)
        n_instance = len(is_success)
        rate_success = n_success / n_instance
        metrics_per_cls[cls_label] = {
            "n_success": n_success,
            "n_instance": n_instance,
            "rate_success": rate_success,
        }
    return metrics_per_cls
