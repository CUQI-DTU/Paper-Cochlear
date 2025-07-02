# Paper-Cochlear

This repository contains code for generating and visualizing the results presented in the paper "Cochlear aqueduct advection and diffusion inferred from computed tomography imaging with a Bayesian approach".

The results for the control experiment are available in the `control_results.ipynb` notebook, while the results for the real data experiment are available in the `real_data_results.ipynb` notebook.

Below is a summary of the command-line arguments used in these two notebooks for running the Bayesian model for ear aqueduct flow model inference.

## Command-Line Arguments for `advection_diffusion_inference.py`

This script runs the Bayesian model for ear aqueduct analysis. It has the following command-line arguments:

### Key Parameters

| Argument | Type | Choices | Description |
|----------|------|---------|-------------|
| `-animal` | `str` | `["m1", "m2", "m3", "m4", "m6"]` | The animal to model (e.g., `"m1"` for mouse 1). |
| `-ear` | `str` | `["l", "r"]` | The ear to model, `'l'` for left or `'r'` for right. |
| `-version` | `str` | — | User defined string used for labeling the experiment. |
| `-sampler` | `str` | `CWMH`, `MH`, `NUTS`, `NUTSWithGibbs` | The MCMC sampler to use. |
| `-unknown_par_type` | `str` | `constant`, `smooth`, `step`, `sampleMean`, `custom_1`, `synth_diff1.npz`, `synth_diff2.npz`, `synth_diff3.npz` | Type of the "true" unknown parameter (diffusion coefficient). |
| `-unknown_par_value` | `str` or list | — | Value(s) of unknown parameter, the diffusion coefficient, if `unknown_par_type` is `constant`, provide one value, if `unknown_par_type` is `step` or `smooth`, provide two values (upper and lower), if `unknown_par_type` is `sampleMean`, provide information about the samples file: tag of the experiment concatenated with the directory name where the samples are stored, separated by `@` |
| `-data_type` | `str` | `real`, `synthetic` | Type of data used for inference, real or synthetic. |
| `-inference_type` | `str` | `constant`, `heterogeneous`, `advection_diffusion` | Type of inference model, `constant` for assuming constant diffusion, `heterogeneous` for heterogeneous diffusion, and `advection_diffusion` for advection-diffusion model. |
| `-Ns` | `int` || Number of MCMC samples to draw. |
| `-Nb` | `int` || Number of burn-in samples. |
| `-noise_level` | `str` || Noise level for data, set to `fromDataVar` to read noise level from data that varies for each data point and set to `fromDataAvg` to compute average noise level from data and use it for all data points, set to `avgOverTime` to compute average noise level over time for each location, set to `estimated` to use the estimated noise level, or set to a float representing the noise level (e.g `0.1` for 10% noise). Noise level can also be a string that starts with `std_` then the std value. For example `std_5` means std of value 5 |
| `-add_data_pts` | `float` list || Additional data points to add (for synthetic cases only). |
| `-num_CA` | `int` | (0–19) | Number of cochlear aqueduct (CA) points to use. |
| `-num_ST` | `int` | (0–8) | Number of scala tympani (ST) points to use. |
| `-true_a` | `float` || True advection speed (regarded in synthetic inference case only). |
| `-rbc` | `str` | `zero`, `fromData`, `fromDataClip` | Right boundary condition. |
| `--adaptive` | `flag` || Use adaptive time stepping (if passed). |
| `-NUTS_kwargs` | `str` || JSON-style string for NUTS sampler options (e.g. `{"max_depth": 10, "step_size": 0.1}`). |
| `--data_grad` | `flag` || If passed, gradient of concentration signal is used as data. |
| `--u0_from_data` | `flag` || If passed, the initial condition is obtained from the data. |
| `--sampler_callback` | `flag` || Enable sampler callback. |
| `--pixel_data` | `flag` || Treat data as pixel-level data. |

**note**: ChatGPT was used to generate first draft of this table based on the provided code snippet of parameter definition and description.
