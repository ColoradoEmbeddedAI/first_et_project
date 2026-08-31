# Installation Guide — ExecuTorch Toolchain Setup

This document covers **one-time environment setup**: installing prerequisites,
cloning and building ExecuTorch for both the host (Linux or macOS) and the
Cortex-M4 target. It is shared across any project in this repo that uses
ExecuTorch — you only need to do this once, then reuse the build for multiple
projects.

Both **Ubuntu 24.04** and **macOS** (13 Ventura or newer, Apple Silicon or
Intel) are supported. Where a step differs, platform-specific blocks are
labelled **Linux** and **macOS**; unlabelled steps are identical on both.

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

### Already set up `first_embedded`?

If you have already successfully built and tested
[ColoradoEmbeddedAI/first_embedded](https://github.com/ColoradoEmbeddedAI/first_embedded)
on this machine, most of this section is done:

- **Ubuntu 24.04:** nothing to do — the ARM toolchain, `openocd`, `cmake`,
  `ninja`, and a compatible Python 3 are all already installed. Skip to
  [Section 2](#2-clone-executorch).
- **macOS:** you only need the Python that ExecuTorch requires:

  ```bash
  brew install python@3.12
  ```

  Then skip to [Section 2](#2-clone-executorch).

If you have **not** done `first_embedded` (or the build failed), follow the
full package list below.

### System packages

#### Linux (Ubuntu 24.04)

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

#### macOS

Install the Xcode Command Line Tools (provides `git`, `make`, `clang`, `xxd`)
and [Homebrew](https://brew.sh), then the build tools and ARM toolchain:

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake ninja python@3.12
brew install --cask gcc-arm-embedded      # arm-none-eabi-gcc/g++/gdb + binutils
brew install open-ocd
```

macOS notes:

- **Python < 3.13 is required for ExecuTorch.** Homebrew's `python` may be
  3.13+, so create the venv with `python3.12` explicitly (see
  [Section 3](#3-build-executorch-for-python)).
- **No `gdb-multiarch` on macOS.** `arm-none-eabi-gdb` (bundled with the
  `gcc-arm-embedded` cask) is the equivalent — substitute it wherever
  [README.md](README.md) says `gdb-multiarch`. On Apple Silicon, `gdb` itself
  is unreliable; you can also debug with the bundled `lldb` attached to
  OpenOCD's gdb server.
- If the `gcc-arm-embedded` cask is blocked by Gatekeeper, run
  `xattr -dr com.apple.quarantine "$(brew --prefix)/Caskroom/gcc-arm-embedded"`,
  or install the Arm GNU Toolchain manually from Arm's site.
- **`nproc` does not exist on macOS.** Each `cmake --build … -j$(nproc)`
  command below is followed by a commented-out `# macOS` line using
  `-j$(sysctl -n hw.logicalcpu)` — run that one instead (plain `-j` also
  works).

### Verify the ARM toolchain

```bash
arm-none-eabi-gcc --version   # should show 13.x or newer
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
    ├── host/                    # ExecuTorch built for the host (Linux/macOS)
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
python3 -m venv et-env        # Linux
# python3.12 -m venv et-env.  # MacOS
source et-env/bin/activate
python --version              # confirm 3.12.x
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

On **macOS** the PyTorch wheels are always CPU/MPS (no CUDA variant exists),
so the `--index-url` is optional — plain
`pip install torch torchvision torchaudio` works too.

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

#### If it Errors on macOS: patch the bundled `flatcc` first, then rerun install_executorch.sh

On recent macOS + Xcode (tested on macOS 15/26 with Apple Clang 17), the
vendored `flatcc` library in `release/1.2` fails to build:

```
flatcc/portable/pprintint.h:388:13: error: implicit conversion loses integer
  precision ... [-Werror,-Wimplicit-int-conversion-on-negation]
grisu3_print.h:186:33: error: initializer-string for character array is too
  long ... [-Werror,-Wunterminated-string-initialization]
make[6]: *** [.../flatccrt.dir/json_printer.c.o] Error 1
```

`install_executorch.sh` then aborts with
`Failed building wheel for executorch`. Newer Clang promotes two warnings to
errors and `flatcc`'s build sets `-Werror`. Disable it in **both** places
ExecuTorch builds `flatcc`, by editing
`~/executorch/executorch/third-party/CMakeLists.txt`:

```cmake
# 1. In the ExternalProject_Add(flatcc_ep ...) CMAKE_ARGS list, add one line:
# (note teh first two lines are for reference to find where to add the needed line)
  CMAKE_ARGS -DFLATCC_RTONLY=OFF
             -DFLATCC_TEST=OFF
             -DFLATCC_ALLOW_WERROR=OFF          # <-- add this
             ...

# 2. Next to the other set(FLATCC_* ... CACHE BOOL "") lines just before
#    add_subdirectory(flatcc), add:
set(FLATCC_ALLOW_WERROR OFF CACHE BOOL "")      # <-- add this
```

Then run `./install_executorch.sh` (Linux users skip straight to it).

> This edits vendored third-party code. `git submodule update` in the
> ExecuTorch tree will revert it — re-apply if that happens. Upstream
> `flatcc` has since fixed this, so a newer ExecuTorch release would not
> need the patch.

If `pip install executorch` above pulled a newer wheel (e.g. `1.4.1`) than
the `release/1.2` source, `install_executorch.sh` replaces it with the
source build (`1.2.0a0`) once it succeeds. Confirm afterwards:

```bash
pip show executorch | grep Version          # expect 1.2.0a0
python -c "from executorch.extension.pybindings import portable_lib"  # no error
```

### Build the ExecuTorch runtime for the host (Linux/macOS)

```bash
source ~/executorch/et-env/bin/activate

cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/host \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,dim_order_ops::_to_dim_order_copy.out"

cmake --build ~/executorch/executorch-build/host -j$(nproc)                       # Linux
# cmake --build ~/executorch/executorch-build/host -j$(sysctl -n hw.logicalcpu)   # macOS
```

The two commands are separate steps:

- `cmake -S … -B …` is only the **configure** step — it finishes in seconds
  to a minute and writes build files. Re-running it is near-instant.
- `cmake --build …` is the **compile** step — this is the one that takes
  15–30 minutes on first run (code generation is slow).

`nproc` is Linux-only; the commented `# macOS` line under each
`cmake --build` uses `sysctl -n hw.logicalcpu` instead. Plain `-j` (no count)
also works on either platform.

#### Harmless "Manually-specified variables were not used" warning

The configure step may end with, e.g.:

```
CMake Warning:
  Manually-specified variables were not used by the project:
    EXECUTORCH_BUILD_KERNELS_PORTABLE
```

This just means a `-D…` flag didn't match an option this ExecuTorch version
defines — usually a flag that was renamed or is only defined when some other
feature is enabled. It is safe to ignore as long as the underlying default
already matches the intent. In `release/1.2` the portable-kernels option is
`EXECUTORCH_BUILD_PORTABLE_OPS` (default `ON`), which is why the commands
above use that name. Confirm the build was configured correctly with:

```bash
cmake -LA -N ~/executorch/executorch-build/host | grep -E \
  'EXECUTORCH_BUILD_PORTABLE_OPS|EXECUTORCH_SELECT_OPS_LIST'
# Expected output includes:
# EXECUTORCH_BUILD_PORTABLE_OPS:BOOL=ON
# EXECUTORCH_SELECT_OPS_LIST:STRING=aten::mul.out,aten::add.out,dim_order_ops::_to_dim_order_copy.out
```

The `EXECUTORCH_SELECT_OPS_LIST` flag lists the ops needed by this project's
model. See [Section 5](#5-rebuilding-after-changing-the-op-set) if your model
needs different ops.

---

## 4. Build ExecuTorch for the Board

### Create the ARM toolchain file

ExecuTorch's release/1.2 branch does not ship a Cortex-M4 toolchain file.
Create it manually.  Note: replace /path/to/first_et_project with the path where the cloned repo [ColoradoEmbeddedAI/first_et_project](https://github.com/ColoradoEmbeddedAI/first_et_project) resides locally:

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
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_SELECT_OPS_LIST="aten::mul.out,aten::add.out,dim_order_ops::_to_dim_order_copy.out" \
  -DMAX_KERNEL_NUM=128 \
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

# Linux
cmake --build ~/executorch/executorch-build/cortex-m4 -j$(nproc) 
# MacOS
cmake --build ~/executorch/executorch-build/cortex-m4 -j$(sysctl -n hw.logicalcpu)   

# Build the selective kernels library (smaller than full portable_ops_lib)
# Linux
cmake --build ~/executorch/executorch-build/cortex-m4 \
  --target executorch_selected_kernels -j$(nproc)               
# MacOS
 cmake --build ~/executorch/executorch-build/cortex-m4 \
   --target executorch_selected_kernels -j$(sysctl -n hw.logicalcpu)    
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

**About `MAX_KERNEL_NUM`:** ExecuTorch's operator registry
(`runtime/kernel/operator_registry.cpp`) reserves a fixed-size table for
registered kernels — `250 operators * 8 kernels/op = 2000` slots by default,
which costs real `.bss` on a Cortex-M4. `-DMAX_KERNEL_NUM=128` shrinks that
table to fit this project's tiny op list, saving RAM. **This flag only
belongs on the `cortex-m4` configure command, never on `host`'s.** The `host`
build whole-archives `portable_ops_lib`, which registers *every* portable op
unconditionally (~200+), regardless of `EXECUTORCH_SELECT_OPS_LIST` — capping
the table at 128 there makes registration overflow and hard-abort before
`main()` ever runs its inference.

Do **not** "fix" this by hand-editing the `kMaxOperators` constant in
`operator_registry.cpp` itself. That file is shared source
compiled into *every* ExecuTorch build tree on this machine (`host`,
`cortex-m4`, and anyone else's) — patching it directly silently breaks every
other tree's host-side inference the next time it's rebuilt, with no record
of why. `MAX_KERNEL_NUM` exists precisely so this can be scoped per build
tree via CMake instead; always use it as a `-D` flag on the specific
`cmake -S ... -B ...` configure command that needs it.

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

# -j$(nproc) below is Linux; on macOS use -j$(sysctl -n hw.logicalcpu) or plain -j

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
