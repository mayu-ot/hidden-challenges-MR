moment_retrieval
==============================

Uncovering Hidden Challenges in Query-Based Video Moment Retrieval

### Dependencies

The code is tested with Python 3.8.

```shell
$ docker build -t hidden-challenges-mr .
```

#### Neptune.ai (optional)

We host our experiments on neptune.ai.
To run output visualization notebooks such as `notebooks/report/2DTAN_ActivityNet.ipynb`, get your API token from [neptune.ai](https://docs.neptune.ai/).

Put your API token in `src/.env` file as:
```:src/.env
NEPTUNE_API_TOKEN="YOUR_TOKEN_HERE"
```

### Data

#### Charades-STA

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

#### ActivityNet Captions
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

### Test blind baselines

```shell
:/app# python src/experiments/blind_baselines.py chrades
:/app# python src/experiments/blind_baselines.py activitynet
```

If this work helps your research, please cite:
```
@article{otani2020challengesmr,
author={Mayu Otani, Yuta Nakahima, Esa Rahtu, and Janne Heikkil{\"{a}}},
title = {Uncovering Hidden Challenges in Query-Based Video Moment Retrieval},
booktitle={The British Machine Vision Conference (BMVC)},
year = {2020},
}
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
