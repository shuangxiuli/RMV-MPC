# Public Release Audit

## Completed

- Repository working directory renamed from `SDP-MPC` to `RMV-MPC`
- Release branch created: `release/rmv-public`
- Public-facing README rewritten for the RMV-MPC release
- Julia project name updated from `DRCC-MPC` to `RMVMPC`
- Old absolute local paths and the `SDP-MPC` identifier removed from the retained official example notebooks
- Notebook outputs cleared from the retained official example notebooks
- Exploratory notebooks, checkpoints, generated gifs, and local analysis artifacts removed from the release working tree
- Git remote `origin` renamed to `upstream` and now points to `labicon/DRCC-MPC`

## Retained notebook allowlist

- `notebook/Eval_Example_1_Synthetic_Gaussian.ipynb`
- `notebook/Eval_Example_2_Data_Trajectron.ipynb`
- `notebook/Eval_Example_3_Data_Gaussian.ipynb`
- `notebook/Eval_Example_4_Data_Oracle.ipynb`
- `notebook/Eval_Example_5_BIC_Synthetic.ipynb`
- `notebook/Eval_Example_6_BIC_Data.ipynb`
- `notebook/Eval_Example_7_Data_Extensive_Search.ipynb`
- `notebook/Eval_Example_8_CrowdNav_Data.ipynb`
- `notebook/Eval_Example_9_DRC_Data_Trajectron.ipynb`
- `notebook/experiment_states.csv`

## Required follow-up before making the GitHub repository public

1. Create a new GitHub repository named `RMV-MPC`
2. Add that repository as the new `origin` remote
3. Review tracked modifications under `src/`, `scripts/`, and the two modified submodules before the first public commit
4. Confirm whether pretrained models and processed datasets are legally redistributable; if not, keep README instructions only
5. Confirm whether any paper-specific results tables should be regenerated instead of committed
6. Add `scripts/default_params/params_synthetic_gaussian.jl` to version control because retained notebooks depend on it

## Remaining legal and provenance checks

- Keep attribution to `labicon/DRCC-MPC`
- Keep attribution to the upstream submodules and their licenses
- Verify whether any third-party model checkpoints should be excluded from the public repository
