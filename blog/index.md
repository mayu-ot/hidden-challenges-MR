---
layout: post
title: Results
subtitle: Our Findings on Query-based Video Moment Retrieval
---

### Blind Baselines Perform Unexpectedly Well
We built three blind baselines which never use videos for training or inference.
Our baselines put scores of deep models in a context.
Surprisingly, our blind baselines are competitive and even outperform some deep models.

- **Prior-Only** : This baseline predicts temporal locations without using videos or query sentences.
- **Action-Aware Blind**: This baseilne uses only one word in a query sentence to predict temporal locations of moments. For simplicity, we use the first verb in a query sentence.
- **Blind-TAN**: This is a neural network-based model that uses the full query sentence. Blind-TAN is built upon 2D-TAN [^1]. Blind-TAN removes a module to extract visual features and replace the module with a learnable map to capture language biases. We trained Blind-TAN to predicts temporal locations solely with query sentences.

[^1]: Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2D temporal adjacent networks for moment localization with natural language. In The AAAI Conference on Artificial Intelligence, 2020.

{% include charades.html %} 
{% include activitynet.html %} 

The score of TripNet is updated according to the latest published scores.

### SOTA Models Often Ignores Visual Input
Our analyses revealed that some deep models highly rely on language priors on video moment retrieval. We describe *visual sanity check* for investigating if a model uses visual input. Visual sanity check is easy to try. We randomly reorder visual features of a video and see how output changes.

![Illustration of sanity check on visual input.]({{site.baseurl}}/img/vis_check.png)

If a model's prediction is based on input videos, the performance should drop significantly. On the other hand, if a model's prediction is mostly based on language priors, the perturbation should hardly affect outputs. Here shows a result from our paper. Except SCDM on Charades-STA, the models highly rely on langauge priors.

![Illustration of sanity check on visual input.]({{site.baseurl}}/img/fig5.png)