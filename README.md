# JMD-FWI

**Joint Model- and Data-Driven Full Waveform Inversion**

> Wu, S., & Geng, J. — *Joint model- and data-driven full-waveform inversion by combining common mid-point gathers and well-log data*, Tongji University.

This repository implements JMD-FWI, a semi-supervised framework that simultaneously integrates:
- 2D acoustic **wave equation** (physics constraint)
- **Common-Shot-Point (CSP) gathers** (standard FWI data domain)
- **Common-Midpoint (CMP) gathers** (neural network input, directly correlatable with 1D well-log data)
- **Migrated seismic data** (structural weighting)
- **Prior well-log data** (labeled supervision)

The key motivation: well-log data is 1D and cannot be directly correlated with 2D/3D CSP gathers. CMP gathers bridge this dimensional gap — each CMP gather naturally captures 1D velocity information below its midpoint. This enables direct embedding of well-log data into the FWI process via neural network training.

---

## Method Overview

### Problem Statement

Conventional FWI is highly nonlinear, depends on a good initial model, and is prone to cycle-skipping when low-frequency seismic data is absent. Directly incorporating prior geological knowledge (well-log data) into the inversion is non-trivial due to the dimensional mismatch between 2D CSP gathers and 1D well logs.

### JMD-FWI Framework

The framework operates in two stages:

**Stage 1 — Build a low-frequency initial model (JMD-FWI):**

A modified U-Net maps 5 consecutive CMP gathers to a 1D vertical velocity profile:

```
v_i = G(d_{i-2}^CMP, d_{i-1}^CMP, d_i^CMP, d_{i+1}^CMP, d_{i+2}^CMP)
```

Training uses a semi-supervised strategy:
- **Labeled data**: CMP gathers at well positions, supervised by low-pass filtered well-log curves
- **Pseudo-labeled data**: CMP gathers at non-well positions, supervised by velocity profiles from conventional FWI on CSP gathers

The total loss function (Huber norm) is:

```
loss_total = λ_l * L_Huber(v_pred_l, v_log) + λ_u * L_Huber(v_pred_u, v_pseudo)
```

where λ_l and λ_u are spatially varying weights computed from the **migrated seismic profile**, which encodes lateral structural continuity.

Once trained, the network inverts all CMP gathers to assemble a full 2D low-frequency velocity model as the initial model for Stage 2.

**Stage 2 — High-resolution FWI:**

Conventional adjoint-state FWI with Adam optimizer and frequency-splitting (starting from the low-frequency band built in Stage 1, then progressively adding higher frequencies). This avoids cycle-skipping by starting from a better initial model.

### Neural Network Architecture

- **Input**: `N_t × N_CMP` CMP gather data (5 traces, limited offset)
- **Feature extraction**: 1D convolutional layers to extract high-dimensional features
- **Encoder-decoder**: U-Net with skip connections and dropout (`Seis_UnetModelDropout` in `nnmodels.py`)
- **Output**: 1D velocity profile (`N_z` depth samples)

---

## Requirements

- Python 3.x
- NumPy, SciPy, Matplotlib
- [PyTorch](https://pytorch.org/) (CUDA GPU strongly recommended; tested on RTX 3080Ti)
- [mpi4py](https://mpi4py.readthedocs.io/) + MPI runtime (e.g., OpenMPI or Intel MPI)
- [Numba](https://numba.pydata.org/) (JIT compilation for finite-difference kernels)
- scikit-image, scikit-learn, tqdm, pandas

```bash
pip install numpy scipy matplotlib torch mpi4py numba scikit-image scikit-learn tqdm pandas
```

---

## Data

The following binary files (float32, C-order, shape `[NZ, NX]`) must be present in the project root:

| File | Shape | Description |
|------|-------|-------------|
| `overthrust_450_175.bin` | [175, 450] | True Overthrust velocity model (m/s) |
| `overthrust_linear_1500_3800.bin` | [175, 450] | Linear gradient initial model (1500–3800 m/s) |
| `over_prior.bin` | [175, 450] | Prior velocity model for regularization |

Read example:
```python
import numpy as np
vp = np.fromfile('overthrust_450_175.bin', dtype=np.float32).reshape(175, 450)
```

The Overthrust model covers a 4.5 km × 1.75 km domain (10 m grid spacing). A water layer is present at the top. One trace at 6000 m depth is used as the pseudo-well log.

---

## Reproduction Steps

### Step 0 — Configure GPU

Edit `seisunet_fwi_split_f_over.py` line 23 and set `CUDA_VISIBLE_DEVICES` to match your GPU:

```python
CUDA_VISIBLE_DEVICES = "0"   # change from default "2"
```

### Step 1 — Pre-compute CMP gather indices

```bash
python cmp.py
```

Generates `cmp_number/{i}_cmp.txt` files (one per CMP location), storing the source-receiver index pairs that contribute to each CMP gather. Required before running inversion.

### Step 2 — Generate synthetic observed seismic data

Skip this step if you already have observed shot records. Otherwise:

```bash
# On a workstation (45 MPI processes = 1 process per shot):
mpiexec -n 45 python mpi_generate_data.py

# On an LSF HPC cluster:
bsub < generate_data.lsf
```

This simulates 45 common-shot-point gathers using the true Overthrust model as the "ground truth" Earth, producing the observed data that FWI will try to match.

### Step 3 — Run the joint inversion

```bash
mpiexec -n <num_procs> python seisunet_fwi_split_f_over.py
```

MPI distributes the 45 shot gathers across processes. Rank 0 handles the neural network training; all ranks participate in wave simulation. After each FWI iteration, gradients are reduced to rank 0 via `MPI_Reduce`, the velocity model is updated, and `MPI_Bcast` redistributes it.

Results are saved to:
```
cmpfwi_over_adam_f_{favg}_linear_model_SNR_{SNR}_lowcut{lowcut_f}Hz/
```

---

## Key Parameters

All editable in `seisunet_fwi_split_f_over.py` (~line 300–350):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NX` / `NZ` | 450 / 175 | Model grid dimensions |
| `DX` / `DZ` / `dt` | 10 m / 10 m / 1 ms | Spatial and temporal sampling |
| `NS` / `NR` | 45 / 450 | Number of shots / receivers |
| `favg` | 10 Hz | Ricker wavelet dominant frequency |
| `Tn` | 3000 | Number of time steps (3 s total) |
| `PML` | 30 | PML absorbing boundary thickness (grid points) |
| `N` | 6 | Finite difference spatial accuracy order |
| `SNR` | 0 | Signal-to-noise ratio (0 = noise-free) |
| `lowcut_f` | 2 Hz | Low-frequency cutoff for frequency-splitting |

### Inversion Strategy (Overthrust)

| Stage | Frequency Band | Purpose |
|-------|---------------|---------|
| JMD-FWI (Stage 1) | 5–6 Hz | Build low-frequency initial model |
| FWI (Stage 2) | 6–20 Hz | High-resolution refinement |

The pseudo-well log (at 6000 m) is 8 Hz low-pass filtered before use as labeled data.

---

## Expected Results

Quantitative metrics (PCC = Pearson Correlation Coefficient, RMSE in m/s) on the Overthrust model:

| Method | RMSE | PCC |
|--------|------|-----|
| Conventional FWI | 320.99 | 0.907 |
| FWI with well-log constraint | 277.26 | 0.931 |
| NN-FWI (Zhu et al., 2022) | 273.12 | 0.939 |
| JMD-FWI (no well-log) | 220.88 | 0.954 |
| **JMD-FWI (with well-log)** | **215.66** | **0.955** |

Conventional FWI fails with cycle-skipping artifacts (falls into local minima). JMD-FWI avoids this by constructing a better low-frequency initial model from CMP gathers and well-log data.

---

## Repository Structure

```
JMD-FWI/
├── seisunet_fwi_split_f_over.py   # Main entry: JMD-FWI pipeline (Overthrust)
├── mpi_generate_data.py            # Generate observed shot records via MPI
├── nnmodels.py                     # Neural network architectures (U-Net variants)
├── cmp.py                          # Pre-compute CMP gather index lookup tables
├── generate_data.lsf               # LSF job submission script for HPC clusters
├── LibConfig.py                    # Library imports
├── ParamConfig.py                  # Network and training hyperparameters
├── PathConfig.py                   # Data and output path configuration
├── func/
│   ├── UnetModel.py                # U-Net encoder-decoder (4 down/up stages, dropout)
│   ├── DataLoad_Train.py           # CMP gather training data loader
│   ├── DataLoad_Test.py            # CMP gather test data loader
│   └── utils.py                    # PSNR, SSIM metrics and utilities
├── cmp_number/                     # CMP index files (generated by cmp.py)
├── overthrust_450_175.bin          # True velocity model [175, 450] float32
├── overthrust_linear_1500_3800.bin # Linear initial model [175, 450] float32
└── over_prior.bin                  # Prior model [175, 450] float32
```

---

## Physics Engine Details

| Component | Implementation |
|-----------|---------------|
| Wave equation | 2D acoustic scalar (eq. 9 in paper) |
| Spatial discretization | 6th-order finite difference (`diff_coef()`) |
| Time integration | 2nd-order leap-frog |
| Absorbing boundary | PML (Perfectly Matched Layer), 30 grid points |
| Source wavelet | Ricker at 10 Hz (`ricker()`) |
| JIT acceleration | Numba `@jit(nopython=True)` on `cal()` and `cal_inverse()` |
| Gradient computation | Adjoint-state method (cross-correlation of forward and adjoint wavefields) |
| Model update | Adam optimizer |

---

## Citation

If you use this code, please cite:

```
Wu, S., & Geng, J. Joint model- and data-driven full-waveform inversion by combining
common mid-point gathers and well-log data. Tongji University.
```

Related references:
- Zhu et al. (2022) — NN-FWI baseline: *Integrating deep neural networks with full-waveform inversion*, Geophysics, 87, R93–R109.
- Asnaashari et al. (2013) — Well-log constrained FWI: *Regularized seismic full waveform inversion with prior model information*, Geophysics, 78, R25–R36.
- Virieux & Operto (2009) — FWI overview: *An overview of full-waveform inversion in exploration geophysics*, Geophysics, 74, WCC1–WCC26.
