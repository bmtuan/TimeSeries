# TUTORIAL

# Model based LSTM combined attention technique

## Install environment from yml file

```bash
cd model
conda env create -f environment.yml
```

## Dataset

There is 8 datasets of 8 difference sensor nodes Fi-Mi. \
Available: **[Dataset](https://github.com/bmtuan/TimeSeries/tree/main/model/multitask_train)**

## Checkpoint

There is full checkpoint of single tasks and multi tasks learning model.
Available: **[Checkpoint](https://github.com/bmtuan/TimeSeries/tree/main/model/best_checkpoint)**

## Usage

To run this code, please follow the instructions below:

### Train

```bash
cd model
multitask_training.py
```

### Test

#### Test for single task

```bash
cd model
demo_singletask.py
```

#### Test for multi task

```bash
cd model
demo_multitask.py
```
