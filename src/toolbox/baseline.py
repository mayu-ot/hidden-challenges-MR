from typing import Tuple, List, Callable
from scipy.stats import gaussian_kde
import numpy as np
from .utils import sentence2token, _nms
from tqdm import tqdm

Query = Tuple[str, str]
Location = Tuple[float, float, float]  # start, end, length
Instance = Tuple[Query, Location]
Rating = List[float]
Prediction = Tuple[Query, List[Location], Rating]
Result = Tuple[Query, List[Location], Rating, dict]


class SegmentGeneratorKDE(object):
    def __init__(self):
        self.kernels = {}
        self.modes = {}
        self.vocab = []

    def fit(self, label, instances):
        start = [location[0] / location[-1] for _, location in instances]
        duration = [
            (location[1] - location[0]) / location[-1]
            for _, location in instances
        ]

        start = np.clip(start, 0, 1)
        duration = np.min(np.vstack((duration, 1 - start)), axis=0)
        print(f"#instances: {len(start)}")
        samples = np.vstack([start, duration])
        self.kernels[label] = gaussian_kde(samples)
        height = self.kernels[label].pdf(samples)
        self.modes[label] = samples[:, np.argmax(height)]

        if label not in self.vocab:
            self.vocab.append(label)

    def sample(self, label, n):
        if label not in self.vocab:
            label = "base"

        samples = self.kernels[label].resample(n)
        likelifood = self.kernels[label](samples)

        start = samples[0, :]
        duration = samples[1, :]

        start = np.clip(start, 0, 1)
        duration = np.clip(duration, 0, 1)

        duration = np.min(np.vstack([duration, 1 - start]), axis=0)

        samples = np.hstack([start[:, None], duration[:, None]])

        return samples, likelifood


def predict(
    segment_generator: SegmentGeneratorKDE,
    instances: List[Instance],
    nms_threshold: float = 0.45,
    top_k: int = 10,
):
    predictions = []
    for instance in tqdm(instances, desc="predicting"):
        query, location = instance
        length = location[-1]

        actions, objects = sentence2token(query[1])

        if len(actions):
            actions = [
                action
                for action in actions
                if action in segment_generator.vocab
            ]
            action = actions[0] if len(actions) else "base"
            samples, likelifood = segment_generator.sample(action, 100)
        else:
            samples, likelifood = segment_generator.sample("base", 100)

        bbox = samples.copy()
        bbox[:, 1] = bbox.sum(axis=1)

        valid = (bbox[:, 1] - bbox[:, 0]) > 0
        bbox = bbox[valid]
        likelifood = likelifood[valid]

        keep = _nms(bbox, likelifood, nms_threshold, top_k=top_k)
        bbox = bbox[keep]
        likelifood = likelifood[keep]

        location = np.hstack([bbox * length, np.ones((len(bbox), 1)) * length])

        predictions.append((query, location.tolist(), likelifood.tolist()))

    return predictions
