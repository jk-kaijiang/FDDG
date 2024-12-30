# Algorithmic Fairness Generalization under Covariate and Dependence Shifts Simultaneously

This repository contains the coded needed to reporduce the results of [Algorithmic Fairness Generalization under Covariate and Dependence Shifts Simultaneously](https://dl.acm.org/doi/pdf/10.1145/3637528.3671909).

In this README, we provide an overview describing how this code can be run.  If you find this repository useful in your research, please consider citing:

```latex
@inproceedings{zhao2024algorithmic,
  title={Algorithmic fairness generalization under covariate and dependence shifts simultaneously},
  author={Zhao, Chen and Jiang, Kai and Wu, Xintao and Wang, Haoliang and Khan, Latifur and Grant, Christan and Chen, Feng},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4419--4430},
  year={2024}
}
```

## Quick Start

### Train a transformation model:

In folder fddg/domainbed/munit/

```python
python train_munit.py --output_path /munit_result_path --input_path1 /munit_result_path/outputs/tiny_munit/checkpoints/ --input_path2 /munit_result_path/outputs/tiny_munit/checkpoints/ --env 0 --device 0 --dataset FairFace --step 12
```

### Train a classifier:

Copy the last checkpoint file in /munit_result_path/outputs/tiny_munit/checkpoints/ to ./domainbed/munit/saved_models/FairFace/0 as 0_cotrain_step1.pt

In folder fddg/

```python
python -m domainbed.scripts.train --output_dir /result_path --test_envs 0 --dataset FairFace
```