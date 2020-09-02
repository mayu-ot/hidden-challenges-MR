hidden-challenges-MR
==============================

Codes of our paper "Uncovering Hidden Challenges in Query-Based Video Moment Retrieval" (BMVC'20)

[[Project page](https://mayu-ot.github.io/hidden-challenges-MR/) | [arXiv](https://arxiv.org/abs/2009.00325) | [YouTube](https://www.youtube.com/watch?v=A_W50Zz6TuE&feature=emb_title) ]

## Dependencies
Docker (recommended)
```shell
$ docker build -t hidden-challenges-mr .
```

or
```shell
$ pip install -r requirements.txt
```
This code is tested with Python3.8

### Neptune.ai (optional)

We host our experiments on neptune.ai.
To run output visualization notebooks such as `notebooks/report/2DTAN_ActivityNet.ipynb`, get your API token from [neptune.ai](https://docs.neptune.ai/).

Put your API token in `src/.env` file as:
```:src/.env
NEPTUNE_API_TOKEN="YOUR_TOKEN_HERE"
```

## Data

### Charades-STA

1. Download [Charades annotations](http://ai2-website.s3.amazonaws.com/data/Charades.zip) and save `Charades_v1_train.csv` and `Charades_v1_test.csv` in `data/raw/charades/`.
2. Download [Charades-STA annotations](https://github.com/jiyanggao/TALL#charades-sta-anno-download). Only train and test annotation files are required.

```
├── data
│   ├── processed
│   └── raw
        └── charades
            └──Charades_v1_train.csv
            └──Charades_v1_test.csv
            └──charades_sta_train.txt
            └──charades_sta_test.txt
```

Then run these commands below:

```shell
$ sh run.sh
:/app# python src/data/make_dataset data/raw/charades/charades_sta_train.txt data/raw/charades/Charades_v1_train.csv
:/app# python src/data/make_dataset data/raw/charades/charades_sta_test.txt data/raw/charades/Charades_v1_test.csv
```

### ActivityNet Captions
Download annotations [here](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip) and save `train.json`, `val_1.json` and `val_2.json` in `data/raw/activitynet/`.

```
├── data
│   ├── processed
│   └── raw
        └── activitynet
            └──train.json
            └──val_1.json
            └──val_2.json
```

## Test blind baselines

```shell
:/app# python src/experiments/blind_baselines.py chrades
:/app# python src/experiments/blind_baselines.py activitynet
```

## Evaluate your model's outputs

`src/toolbox` provides tools for evaluation and visualization of moment retrieval.
For example, evaluation on Charades-STA is done as:

```python
from src.toolbox.data_converters import CharadesSTA2Instances
from src.toolbox.eval import evaluate, accumulate_metrics

test_data = CharadesSTA2Instances(
    pd.read_csv(f"data/processed/charades/charades_test.csv")
)
############################
## your prediction code here
## ....
############################

results = evaluate(test_data, predictions)
summary = accumulate_metrics(results)
```
`predictions` is a list of model's output.
Each item should be in the format as:
```
(
 (video_id: str, description: str),
 List[(moment_start: float, moment_end: float, video_duration: float)],
 List[rating: float]
)
```
- `video_id`: video ID
- `description`: a query sentence. 
- `moment_start`: a starting point of predicted moment's location in seconds
- `moment_end`: a end point of predicted moment's location in seconds
- `video_duration`: the duration of a whole video in seconds.
- `rating`: a score of a predicted location. A prediction with the largest `rating` is evaluated as top-1 prediction.

For example, an item in `predictions` is like:
```
predictions[0]

(('3MSZA', 'person turn a light on.'),
 [[0.76366093268685, 7.389522474042329, 30.96],
  [21.86557223053205, 29.71737331263709, 30.96],
  ...
  ],
 [7.252954266982226,
  4.785879048072588,
  ...])
```

`summary` is a dictionary of metrics (R@k (IoU>m)).
Examples of how to use our toolbox are in `src/experiments/blind_baselines.py` or notebooks (e.g., 
`notebooks/report/SCDM_CharadeSTA.ipynb`).

If this work helps your research, please cite:
```
@inproceedings{otani2020challengesmr,
author={Mayu Otani, Yuta Nakahima, Esa Rahtu, and Janne Heikkil{\"{a}}},
title = {Uncovering Hidden Challenges in Query-Based Video Moment Retrieval},
booktitle={The British Machine Vision Conference (BMVC)},
year = {2020},
}
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
