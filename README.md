moment_retrieval
==============================

Uncovering Hidden Challenges in Moment Retrieval

### Dependencies

```shell
$ cd docker
$ docker build -t hidden-challenges-mr .
```

### Data

Download annotation files of Charades-STA and ActivityNet Captions.

```shell
$ sh run.sh
:/app# python src/data/make_dataset data/raw/charades/charades_sta_train.txt data/raw/charades/Charades_v1_train.csv
:/app# python src/data/make_dataset data/raw/charades/charades_sta_test.txt data/raw/charades/Charades_v1_test.csv
```

### Test blind baselines

```shell
:/app# python src/experiments/blind_baselines.py chrades
:/app# python src/experiments/blind_baselines.py activitynet
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
