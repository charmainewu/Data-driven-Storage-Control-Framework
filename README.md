# Data-driven-Storage-Control-Framework

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Overview](#overview)
- [Maintainers](#maintainers)

## Background

Dynamic pricing is both an opportunity and a challenge to the end-users. It is an opportunity as it better reflects the real-time market conditions and hence enables an active demand side. However, demand's active participation does not necessarily lead to benefits. The challenge conventionally comes from the limited flexible resources and limited intelligent devices on the demand side. The decreasing cost of the storage system and the widely deployed smart meters inspire us to design a data-driven storage control framework for dynamic prices. Our work first establishes a stylized model by assuming the knowledge on the structure of dynamic price distributions and designs the optimal storage control policy. Based on Gaussian Mixture Model, we propose a practical data-driven control framework, which helps relax the assumptions in the stylized model. Numerical studies illustrate the remarkable performance of the proposed data-driven framework.

## Requirements

The following requirements should be satisfied:

```sh
Python==3.6
tensorflow==1.14.0
Kera==2.2.4
scikit-learn==0.22.1
matplotlib==3.1.2
seaborn==0.9.0
pandas==0.25.3
numpy==1.16.4
```

## Overview
### Data traces
The prices, load as well as the renewable generation data traces are plotted.
### One-shot load serving problem
This part of code plots the performance of DETA using the synthetic data, which displays diminishing regret ratio, and implies DETA converges to the offline optimal rather fast in the one-shot load serving problem.
### General load serving problem
The performacne of general load serving problem is evaluated by synthetic data and real data. For each data we plot the empirical regret ratio, decision details as well as the cost in absolute value of the proposed framework and benchmarks. 
### Case Study
The performance of heuristics is evalueted by real data of each month. We plot the empirical regret ratio and the cost in absolute value of the proposed framework and its variants. 
### Appendix
The length of one-shot laod serving problem is presented.
The renewables generation prediction is presented.
The net marginal benefit of the proposed framework considering charging/discharging efficiency, degredation as well as the leakage is presented.

## Maintainers

[@charmainewu](https://github.com/charmainewu).

