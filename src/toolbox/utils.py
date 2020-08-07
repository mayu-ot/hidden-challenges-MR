from typing import List, Tuple
import numpy as np
import spacy

_nlp = spacy.load("en_core_web_sm")


def _load_top_actions(data_name="charades"):
    return [
        line.rstrip()
        for line in open(f"data/processed/{data_name}/top50_actions")
    ]


def sentence2token(sentence: str) -> Tuple[List[str], List[str]]:
    tokens = _nlp(sentence)

    verb_word = []
    obj_word = []

    for token in tokens:
        if token.is_stop:
            continue

        if token.text == "person":
            continue

        if token.tag_[:2] == "VB":
            verb_word.append(token.lemma_)

        if token.tag_ in ["NN", "NNS"]:
            obj_word.append(token.lemma_)

    key_tokens = (verb_word, obj_word)

    return key_tokens


def _nms(dets: np.ndarray, scores: np.ndarray, thresh: float = 0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = scores
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
