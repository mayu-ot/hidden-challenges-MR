---
layout: post
title: How Well are the Blind Baselines?
date: 2020-09-1
categories: tldr
image: https://raw.githubusercontent.com/mayu-ot/hidden-challenges-MR/gh-pages/img/charades-leaderboard.png
---

#### Blind Baselines Perform Unexpectedly Well
We built three blind baselines which never use videos for training or inference.
Our baselines put scores of deep models in a context.
Surprisingly, our blind baselines are competitive and even outperform some deep models.

- **Prior-Only Blind** : This baseline predicts temporal locations without using videos or query sentences.
- **Action-Aware Blind**: This baseilne uses only one word in a query sentence to predict temporal locations of moments. For simplicity, we use the first verb in a query sentence.
- **Blind-TAN**: This is a neural network-based model that uses the full query sentence. Blind-TAN is built upon 2D-TAN [^1]. Blind-TAN removes a module to extract visual features and replace the module with a learnable map to capture language biases. We trained Blind-TAN to predicts temporal locations solely with query sentences.

[^1]: Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2D temporal adjacent networks for moment localization with natural language. In The AAAI Conference on Artificial Intelligence, 2020.

{% include charades.html %} 
{% include activitynet.html %} 

The score of TripNet is updated according to the latest published scores.