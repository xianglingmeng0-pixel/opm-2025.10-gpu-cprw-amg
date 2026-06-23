#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/opm-simulators-gpu-cprw-paper}
GPU_BUILD=${GPU_BUILD:-/workspace/opm-2025.10/src/opm-simulators/build-amgx}
CPU_BUILD=${CPU_BUILD:-/workspace/opm-2025.10/src/opm-simulators/build-hypre}
CASE=${CASE:-/workspace/cases/opm-data-sleipner/sleipner/SLEIPNER_ORG.DATA}
SOLVERS=$ROOT/paper-artifacts/solver-json
LOGDIR=${LOGDIR:-/workspace}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

cd "$GPU_BUILD"
export OMP_NUM_THREADS=${GPU_THREADS:-12}

./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=gpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_gpuistl_ilu0_tol005_max200.json" \
  | tee "$LOGDIR/log_sleipner_gpuilu0_tol005_max200_1mpi12t_full.log"

./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=gpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_gpuistl_dilu_tol005_max200.json" \
  | tee "$LOGDIR/log_sleipner_gpudilu_tol005_max200_1mpi12t_full.log"

./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=gpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_amgx_gpu_cpr_stable.json" \
  | tee "$LOGDIR/log_sleipner_gpu_cpr_1mpi12t.log"

./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=gpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_amgx_gpu_cprw_fullw_stable.json" \
  | tee "$LOGDIR/log_sleipner_gpu_cprw_1mpi12t.log"

cd "$CPU_BUILD"
export OMP_NUM_THREADS=1

mpirun --allow-run-as-root \
  --mca pml ob1 \
  --mca btl self,tcp \
  -np ${CPU_MPI:-8} \
  ./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=cpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_cpu_cpr_dune_unified.json" \
  | tee "$LOGDIR/log_sleipner_cpu_cpr_8mpi1t.log"

mpirun --allow-run-as-root \
  --mca pml ob1 \
  --mca btl self,tcp \
  -np ${CPU_MPI:-8} \
  ./bin/flow "$CASE" \
  --parsing-strictness=low \
  --check-satfunc-consistency=false \
  --linear-solver-accelerator=cpu \
  --matrix-add-well-contributions=true \
  --full-time-step-initially=1 \
  --linear-solver="$SOLVERS/solver_cpu_cprw_dune_unified.json" \
  | tee "$LOGDIR/log_sleipner_cpu_cprw_8mpi1t.log"
