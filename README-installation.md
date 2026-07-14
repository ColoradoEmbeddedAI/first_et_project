# Installation Guide — ExecuTorch Toolchain Setup

This document covers **one-time environment setup**: installing prerequisites,
cloning and building ExecuTorch for both the Linux host and the Cortex-M4
target. It is shared across any project in this repo that uses ExecuTorch —
you only need to do this once, then reuse the build for multiple projects.

For project-specific usage (exporting the model, building/running
`first_et_project` itself), see [README.md](README.md).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone ExecuTorch](#2-clone-executorch)
3. [Build ExecuTorch for Python](#3-build-executorch-for-python)
4. [Build ExecuTorch for the Board](#4-build-executorch-for-the-board)
5. [Rebuilding After Changing the Op Set](#5-rebuilding-after-changing-the-op-set)

---

## 1. Prerequisites

### System packages

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 python3-pip python3-venv \
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    gdb-multiarch \
    openocd \
    xxd
```

Note: Python < 3.13 is required for ExecuTorch. Ubuntu 24.04 defaults to a
compatible version; Ubuntu 26.04 might not.

Verify the ARM toolchain:

```bash
arm-none-eabi-gcc --version   # should show 13.x
openocd --version             # should show 0.12.x
```

### Hardware

- STM32F407VG Discovery board (or any STM32F407 board)
- ST-Link debugger (built into the Discovery board)
- USB cable

### Directory layout this guide produces

```
~/executorch/
├── executorch/                  # ExecuTorch source (git clone)
├── et-env/                      # Python virtual environment
└── executorch-build/
    ├── host/                    # ExecuTorch built for Linux
    └── cortex-m4/               # ExecuTorch built for STM32
```

---

## 2. Clone ExecuTorch

This guide targets **release/1.2 (v1.2.0)**, the version validated against
this project. Using a different version may require re-verifying the
bare-metal gotchas documented in [README.md](README.md).

```bash
mkdir -p ~/executorch
cd ~/executorch

git clone -b release/1.2 https://github.com/pytorch/executorch.git executorch
cd executorch

# Initialize ALL submodules — missing submodules cause CMake errors
# like "does not contain a CMakeLists.txt file" for flatbuffers, gflags, json
git submodule sync
git submodule update --init --recursive
```

Note: you may see a pip warning like "ERROR: pip's dependency resolver does
not currently take into account...". This is expected and can be ignored.

---

## 3. Build ExecuTorch for Python

### Create the virtual environment

```bash
cd ~/executorch
python3 -m venv et-env
source et-env/bin/activate
```

When the venv is active, your prompt is prefixed with its name, e.g.:

```
(et-env) erke9581@ECEE-FAC-L-028:~/executorch$
```

> **Important:** Always activate the venv before running cmake or python
> scripts: `source ~/executorch/et-env/bin/activate`
>
> CMake's configure step calls Python to run ExecuTorch's code generation
> tools (`gen_oplist`, etc.). If it finds the system Python instead of the
> venv Python, you get `ModuleNotFoundError: No module named 'executorch'`.

### Install PyTorch and ExecuTorch

```bash
# Install CPU-only PyTorch — the CUDA wheel causes CMake errors on
# machines without CUDA because Caffe2's CMake config requires CUDA
# if the wheel was built with it.
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

pip install executorch
```

### Build the ExecuTorch Python package

```bash
cd ~/executorch/executorch
source ~/executorch/et-env/bin/activate

./install_executorch.sh

# This was needed due to the system getting confused with cmake
# (on host it was installed, in venv it gets installed again).
# This makes sure we use the one inside the venv. No harm doing it.
hash -r
```

### Build the ExecuTorch runtime for the Linux host

```bash
source ~/executorch/et-env/bin/activate

cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/host \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_PORTABLE=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,dim_order_ops::_to_dim_order_copy.out"

cmake --build ~/executorch/executorch-build/host -j$(nproc)
```

This takes 15–30 minutes on first run (code generation is slow).

The `EXECUTORCH_SELECT_OPS_LIST` flag lists the ops needed by this project's
model. See [Section 5](#5-rebuilding-after-changing-the-op-set) if your model
needs different ops.

---

## 4. Build ExecuTorch for the Board

### Create the ARM toolchain file

ExecuTorch's release/1.2 branch does not ship a Cortex-M4 toolchain file.
Create it manually:

```bash
mkdir -p ~/executorch/executorch/cmake/toolchains
cp /path/to/first_et_project/arm-none-eabi.cmake \
   ~/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake
```

Or create it with this content:

```cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_C_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mno-unaligned-access -fno-pic -fno-pie")
set(CMAKE_CXX_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mno-unaligned-access -fno-pic -fno-pie")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

### Configure and build

```bash
cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/cortex-m4 \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_HOST_TARGETS=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_KERNELS_PORTABLE=ON \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,dim_order_ops::_to_dim_order_copy.out" \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF \
  -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=OFF \
  -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_CPUINFO=OFF \
  -DPTHREADPOOL_ALLOW_DEPRECATED_API=OFF \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_THREADPOOL=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF

cmake --build ~/executorch/executorch-build/cortex-m4 -j$(nproc)

# Build the selective kernels library (smaller than full portable_ops_lib)
cmake --build ~/executorch/executorch-build/cortex-m4 \
  --target executorch_selected_kernels -j$(nproc)
```

### Flag reference

**Why so many `OFF` flags?** XNNPACK, pthreadpool, and cpuinfo all have
pthread dependencies that don't exist on bare-metal. They are enabled by
default in ExecuTorch and must be explicitly disabled:

```bash
-DEXECUTORCH_BUILD_XNNPACK=OFF
-DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF
-DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=OFF
-DEXECUTORCH_BUILD_PTHREADPOOL=OFF
-DEXECUTORCH_BUILD_CPUINFO=OFF
-DPTHREADPOOL_ALLOW_DEPRECATED_API=OFF
-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF
-DEXECUTORCH_BUILD_EXTENSION_THREADPOOL=OFF
```

After configuring, verify no XNNPACK components are still `ON`:

```bash
grep -i "xnnpack\|pthreadpool\|cpuinfo\|kleidi" \
  ~/executorch/executorch-build/cortex-m4/CMakeCache.txt | grep "=ON"
```

Should return nothing. If it doesn't, the build will fail with
`pthread_once_t`, `pthread_mutex_t`, or `pthread_t` errors.

**About `EXECUTORCH_SELECT_OPS_LIST`:** The ops listed here depend on your
model. Find which ops your model needs from the export script output (see
`models/export_model.py`). Use the `.out` variant names (e.g. `aten::mul.out`,
not `aten::mul.Tensor`) — ExecuTorch's portable kernels register under the
`.out` variants.

**`EXECUTORCH_SELECT_OPS_LIST` does NOT affect `portable_ops_lib`.** It only
affects `executorch_selected_kernels`. Always build and link
`executorch_selected_kernels` (not `portable_ops_lib`) for size-constrained
targets — the full `portable_ops_lib` with all ops exceeds 10MB and won't fit
in 1MB flash.

**`$HOME` vs `~`:** CMake does not expand `~` in `-D` arguments — it's shell
shorthand, and CMake receives it literally, producing errors like
`Could not find toolchain file: ~/executorch/...`. Use `$HOME` (expanded by
the shell) or a full absolute path instead. `CMakePresets.json` uses
`$env{HOME}`, the presets-syntax equivalent.

---

## 5. Rebuilding After Changing the Op Set

If you change the model (in a downstream project) such that it needs
different operators, you must update `EXECUTORCH_SELECT_OPS_LIST` and
reconfigure/rebuild **both** ExecuTorch build trees — the flag is baked into
generated code at configure time, so editing it requires a fresh `cmake`
configure, not just a rebuild.

```bash
source ~/executorch/et-env/bin/activate

# 1. Re-configure and rebuild the host tree with the new op list
cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/host \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,<...new ops...>" \
  # ...(all other flags from Section 3 unchanged)
cmake --build ~/executorch/executorch-build/host -j$(nproc)

# 2. Re-configure and rebuild the cortex-m4 tree with the new op list
cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/cortex-m4 \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,<...new ops...>" \
  # ...(all other flags from Section 4 unchanged)
cmake --build ~/executorch/executorch-build/cortex-m4 -j$(nproc)
cmake --build ~/executorch/executorch-build/cortex-m4 \
  --target executorch_selected_kernels -j$(nproc)
```

Then rebuild the project itself (see [README.md](README.md)) so it links
against the updated `executorch_selected_kernels`.

**Verify the new ops actually registered:**

```bash
grep "Kernel(" ~/executorch/executorch-build/cortex-m4/executorch_selected_kernels/RegisterCodegenUnboxedKernelsEverything.cpp
```

You should see one `Kernel(` line per selected op. If an op is missing, check
that you used the `.out` variant name.
