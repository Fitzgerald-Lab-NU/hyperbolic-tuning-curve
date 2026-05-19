  <!-- <a href="LICENSE">
    <img src="https://img.shields.io/github/license/HazyResearch/bwler" alt="License"/>
  </a> -->
  <!-- a href="https://arxiv.org/abs/2506.23024">
    <img src="https://img.shields.io/badge/arXiv-2506.23024-b31b1b.svg" alt="arXiv"/> 
  </a> -->

<div align="center">

<h1 align="center" style="font-size: 28px;">Hyperbolic neural population geometry benefits computation </h1>

<p align="center">
  This repository contains code for the following paper:
</p>

<blockquote align="center">
  <b>Hyperbolic neural population geometry benefits computation</b><br/>
  Dennis Wu, Yi-Chun Hung, Braden Yuille, James E. Fitzgerald*, Han Liu*<br/>
  The International Conference on Machine Learning (ICML) 2026 <br/>
  <small><a href="https://openreview.net/forum?id=WXNjDNDnpy"><em>[Read the paper]</em></a></small>
</blockquote>

</div>

## Abstract 

Neural population geometry shapes downstream computation. 
Recent empirical findings in neurobiology suggest that a hyperbolic structure underlies population activity in the hippocampus.
Here we provide a theoretical framework for this phenomenon. 
First, we propose a plausible construction of hippocampal tuning curves that statistically induces hyperbolic geometry. 
Next, we establish a connection between neural decoding and associative memory by demonstrating that the Modern Hopfield Network update rule computes the minimum mean-squared-error (MMSE) estimator.
Finally, we introduce a novel associative memory model defined in hyperbolic space that yields significantly larger capacity than leading models. 
Our results suggest that animals encode spatial information as a latent hyperbolic cognitive map, improving both memory capacity and decoding accuracy.

<div align="center">
  <img src="imgs/Group 4.png" alt="overview" height="400"/>
  <br/>
</div>

<br>

<p>
  This paper shows that under <b>exponentially distributed place field sizes</b>, the population geometry induced by the hippocampal place cells is hyperbolic.
</p>

<p>
  This repository provides implementation for:
</p>

<ul>
  <li><b>Karcher-flow Model:</b> A computational model of associative memory that operates on the hyperboloid model. </li>
  <li><b>Karcher-flow layers:</b> Hyperbolic machine learning layers inspired by the Karcher-flow model.</li>
  <li><b>Hyperbolicity of Gaussian tuning:</b> A tutorial of estimating hyperbolicity for tuning curves under different place field size distributions.</li>
</ul>


### Dependencies

From the **`ok1/`** directory:

```bash
uv venv
source .venv/bin/activate  
uv sync
uv pip install -e .
```


### Code structure

The codebase is organized as follows:

- [scripts/](scripts/): contains scripts for running the experiments:
  - [scripts/capacity_simulation.py](scripts/capacity_simulation.py): capacity simulation scripts
  - [scripts/inclass_simulation.py](scripts/inclass_simulation.py): capacity simulation within similar patterns scripts
- [src/experiments/cli](src/experiments/cli): main experiment interface
- [src/experiments/recall](src/experiments/recall): implementation of the Karcher-flow model and other baselines
- [src/experiments/config](src/experiments/config): contains configs of the recall simulation
- [notebooks/Statistically_hyperbolic.ipynb](notebooks/Statistically_hyperbolic.ipynb): simulations for statistically hyperbolic semi-metric spaces.


### Quick start

<br>

#### Statistically hyperbolic semi-metric space


We provide a short tutorial on verifying whether a semi-metric space is statistically hyperbolic.
See `notebooks/Statistically_hyperbolic.ipynb` for more details.
The results show that both the exponential distribution and the log-normal distribution are able to induce a semi-metric space that is statistically hyperbolic, which are the two place field size distributions reported in [Zhang et al. 2022](https://www.nature.com/articles/s41593-022-01212-4#Sec8).

<div align="center">
  <img src="imgs/stat-hyp.png" alt="overview" height="340"/>

  <br/>

</div>



#### Capacity simulation

Using cuda is highly recommended for simulations

```bash
python capacity_simulation.py --M-min 20 --M-max 200 --M-step 20 --n-trials 10 --max-steps 5

python capacity_simulation.py --dataset mnist --no-pca --device cuda \
  --M-min 10 --M-max 400 --n-trials 5 --max-steps 5 --mem-R 3
```

Image runs write under `outputs/<dataset>/<pixels|pca{d}>/Radius<R>/beta<β>/` (e.g. `beta10`, `beta1`).


####  In-class simulation

Outputs: `outputs/<dataset>/inclass/class<id>/<pixels|pca{d}>/Radius<R>/beta<β>/`.

```bash
python3 inclass_simulation.py --dataset cifar10 --class-id 3 --pca-dim 50 \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cpu

python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda
```

To run all simulations, see `run_exp.sh`


### Cite

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{
anonymous2026hyperbolic,
title={Hyperbolic neural population geometry benefits computation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=WXNjDNDnpy}
}
```



salloc --account=p32593 --job-name=ok --nodes=1 --partition=gengpu --gres=gpu:a100:1 --ntasks-per-node=1 --cpus-per-task=16 --mem=80G --time=01:00:00

srun --jobid=8118904 --pty bash -l



python3 capacity_simulation.py --dataset cifar10 --device cuda --no-pca \
  --M-min 100 --M-max 200 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "A" \
  --wandb-tags "sim:capacity,dataset:cifar10,feat:pixels,R:2,device:cuda"

