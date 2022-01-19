# A Cost-Effective Framework for Preference Elicitation and Aggregation
This repo reproduces the paper [A Cost-Effective Framework for Preference Elicitation and Aggregation](https://arxiv.org/abs/1805.05287). The paper appeared in [UAI 2018](https://auai.org/uai2018/) and [WADE 2018](https://sites.google.com/view/wade-workshop/). This repo requires Python 2.7.

To generate the synthetic data:
```
python generateDecisions.py
```

To run the experiment with the synthetic data:
```
python main.py proposed random d-opt e-opt
```
