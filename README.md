# GPU CPRW-AMG for OPM Flow 2025.10

This repository contains the artifact for the paper:

**A GPU-Accelerated CPRW-AMG Preconditioner for Porous Media Reservoir Simulation**

The code extends the GPU-ISTL solver path in OPM Flow 2025.10 with a GPU CPRW-AMG preconditioner. The implementation extends the existing GPU CPR-AMG pressure-only coarse system to a pressure--BHP coarse system and updates well-related coarse entries using sparse GPU kernels.

## Main Contributions

The artifact includes:

- a GPU pressure--BHP transfer policy for CPRW-AMG;
- GPU sparse set/add kernels for well-related coarse entries;
- integration of `type: cprw` into the GPU-ISTL preconditioner factory;
- solver JSON files and scripts used for the paper experiments;
- processed benchmark results for SPE9, SPE10, Norne, and Sleipner.

The artifact branch intentionally excludes the experimental native GPU well-operator path. The validated paper experiments use:

```bash
--linear-solver-accelerator=gpu
--matrix-add-well-contributions=true





```

## Artifact Contents

```text
paper-artifacts/
  solver-json/      Solver configurations used in the experiments
  scripts/          Reproduction scripts
  results/          Processed result tables in CSV format
```

## Reproducing Experiments

```bash
cd /workspace/opm-simulators-gpu-cprw-paper

export ROOT=/workspace/opm-simulators-gpu-cprw-paper
export CUDA_VISIBLE_DEVICES=1

paper-artifacts/scripts/run_spe9.sh
paper-artifacts/scripts/run_norne.sh
paper-artifacts/scripts/run_spe10.sh
paper-artifacts/scripts/run_sleipner.sh
```

Large cases such as SPE10 and Sleipner can take many hours or days for the single-level GPUILU0/GPUDILU baselines.

To summarize logs:

```bash
paper-artifacts/scripts/summarize_log.sh /workspace/log_spe9_gpu_cprw_1mpi12t.log
```

## Results

Processed results are stored in:

```text
paper-artifacts/results/spe9_results.csv
paper-artifacts/results/norne_results.csv
paper-artifacts/results/spe10_results.csv
paper-artifacts/results/sleipner_results.csv
```

The `linear_apply_s` column is computed as:

```text
linear_apply_s = linear_solve_s - linear_setup_s
```

The `assembly_update_s` column is computed as:

```text
assembly_update_s = assembly_time_s + props_update_time_s
```

## License

This artifact is based on OPM Flow and follows the license terms of the OPM project.

## Implementation Scope

This artifact implements the stable GPU CPRW-AMG path used in the paper experiments. The implementation extends the existing GPU CPR-AMG infrastructure in OPM Flow 2025.10 from a pressure-only coarse system to a pressure--BHP coarse system.

The validated paper configuration uses assembled well contributions:

```bash
--matrix-add-well-contributions=true
```

In this configuration, well contributions are included consistently in the matrix used by the Krylov operator, the GPU fine-level smoother, and the CPRW pressure--BHP coarse matrix construction. This is the stable path used for the SPE9, SPE10, Norne, and Sleipner results.

The experimental native GPU well-operator path, corresponding to:

```bash
--matrix-add-well-contributions=false
```

is not included in this artifact branch. That path was used only for exploratory debugging and is left as future work because it requires separate equivalence validation of the GPU well matrix-vector product.

## Modified Source Files

The main source files changed or added by this artifact are:

- `opm/simulators/linalg/gpuistl/GpuPressureBhpTransferPolicy.hpp`
- `opm/simulators/linalg/gpuistl/GpuCprwWellContext.hpp`
- `opm/simulators/linalg/StandardPreconditioners_gpu_serial.hpp`
- `opm/simulators/linalg/gpuistl/ISTLSolverGPUISTL.hpp`
- `opm/simulators/linalg/gpuistl/PreconditionerFactory_gpu_instantiate.cpp`
- `opm/simulators/linalg/gpuistl/detail/cpr_amg_operations.hpp`
- `opm/simulators/linalg/gpuistl/detail/cpr_amg_operations.cu`

`GpuPressureBhpTransferPolicy.hpp` extends the GPU coarse transfer from pressure-only unknowns to pressure--BHP unknowns. `GpuCprwWellContext.hpp` provides well-related coarse-system data to the GPU transfer policy. The CUDA kernels in `cpr_amg_operations.cu` add mapped coarse-entry construction and sparse set/add operations for well-related coarse entries. The GPU preconditioner factory is extended so that JSON files with:

```json
"type": "cprw"
```

enter the GPU-ISTL CPRW-AMG path.

## Solver Backends

The paper comparisons use matched solver configurations as far as possible.

GPU CPR-AMG and GPU CPRW-AMG use:

- BiCGSTAB through the GPU-ISTL solver path;
- GPU DILU as the fine-level smoother;
- AMGX as the coarse AMG backend;
- identical Krylov tolerances, outer smoothing parameters, and AMGX settings.

CPU CPR-AMG and CPU CPRW-AMG use:

- BiCGSTAB through the CPU ISTL solver path;
- DUNE AMG as the coarse AMG backend;
- matched CPR/CPRW parameters where applicable.

GPUILU0 and GPUDILU are included as single-level GPU baselines.

## Runtime Environment

The experiments were run in an Ubuntu 22.04 based CUDA container with:

- OPM Flow 2025.10
- CUDA 12.4
- NVIDIA driver 550.127.05
- NVIDIA GeForce RTX 4090 GPU
- Intel Xeon Gold 6230R CPU
- AMGX coarse AMG backend for GPU CPR-AMG and GPU CPRW-AMG
- DUNE AMG backend for CPU CPR-AMG and CPU CPRW-AMG

The benchmark scripts in `paper-artifacts/scripts/` set the MPI and OpenMP configurations used in the paper tables. The processed results are stored in `paper-artifacts/results/`.

## Artifact Contents

This artifact intentionally includes only lightweight reproducibility files:

- source-code changes;
- solver JSON files;
- run scripts;
- processed result CSV files;
- README and citation metadata.

It does not include build directories, generated simulator output files, restart files, or large reservoir output files.
