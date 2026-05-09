This repository implements our algorithm.

env_back.yml includes conda env for this repo.
``` Bash
# create conda environment
conda env create -f env_back.yml
```

custom_envs includes modified environments for training.

please use commands below for training:
```
python scripts/train_multi_pri.py
```
```
python scripts/train_mpe_pri.py
```


