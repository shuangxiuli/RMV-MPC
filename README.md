# RMV-MPC

Official code release for the RMV-MPC method in our IROS paper on robot navigation under uncertainty.

This repository is derived from [labicon/DRCC-MPC](https://github.com/labicon/DRCC-MPC) and retains the original third-party integrations needed for reproduction, including Trajectron++, CrowdNav, and Python-RVO2 through git submodules.

## Scope

This public release is organized as a reproducible research codebase rather than a polished software package. The main contents are:

- Julia source for the RMV-MPC controller and evaluation pipeline
- Example notebooks for the paper experiments
- Parameter setup scripts and test cases
- Git submodule links for third-party dependencies

## Environment

Tested environment:

- Ubuntu 20.04 (WSL2 is acceptable)
- ROS Noetic
- Julia 1.7.3
- Conda Python 3.6 environment for Trajectron++ and CrowdNav

## Clone

Clone with submodules:

```bash
git clone --recurse-submodules <your-repo-url>
cd RMV-MPC
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Python environment

Trajectron++ depends on Python 3.6 in this codebase.

```bash
conda create -n rmvmpc python=3.6 -y
conda activate rmvmpc
```

Install Trajectron++ dependencies:

```bash
cd Trajectron-plus-plus
pip install -r requirements.txt
```

Install Python-RVO2:

```bash
cd ../Python-RVO2
pip install -r requirements.txt
python setup.py build
python setup.py install
```

Install CrowdNav:

```bash
cd ../CrowdNav
pip install -e .
```

## Julia environment

From the repository root:

```bash
julia --project=.
```

Then instantiate the Julia environment:

```julia
using Pkg
Pkg.instantiate()
```

## Models and datasets

This repository does not assume that all pretrained models or processed datasets can be redistributed here.

You may need to place the following assets manually in the paths expected by the notebooks and scripts:

- processed Trajectron++ pedestrian datasets under `Trajectron-plus-plus/experiments/processed/`
- pretrained Trajectron++ checkpoints under `Trajectron-plus-plus/experiments/pedestrians/models/`
- any additional CrowdNav model files used by the corresponding example notebooks

## Main entry points

Primary example notebooks are under `notebook/`:

- `Eval_Example_1_Synthetic_Gaussian.ipynb`
- `Eval_Example_2_Data_Trajectron.ipynb`
- `Eval_Example_3_Data_Gaussian.ipynb`
- `Eval_Example_4_Data_Oracle.ipynb`
- `Eval_Example_5_BIC_Synthetic.ipynb`
- `Eval_Example_6_BIC_Data.ipynb`
- `Eval_Example_7_Data_Extensive_Search.ipynb`
- `Eval_Example_8_CrowdNav_Data.ipynb`
- `Eval_Example_9_DRC_Data_Trajectron.ipynb`

## Acknowledgement

This repository builds on:

- `labicon/DRCC-MPC`
- `StanfordMSL/RiskSensitiveSAC.jl`
- `StanfordASL/Trajectron-plus-plus`
- `vita-epfl/CrowdNav`
- `sybrenstuvel/Python-RVO2`

Please preserve their licenses and citations when reusing this code.

## License

This repository remains under the MIT License. See `LICENSE`.
