# 0004: Which ONNX Runtime build the GPU pack runs

- Status: accepted, 2026-09-26 (issue #10)
- Applies to: the CUDA pack (`lib/ollaya/cuda_v13`), `ollaya-cuda-runner`, the `:cuda` image. The
  CPU build and the macOS builds are unchanged, and CPU runners keep using the CPU build.

## Context

`ort` 2.0.0-rc.13 (pinned exactly in `Cargo.toml`) downloads pyke's ONNX Runtime 1.28.0 and links
it statically into `bin/ollaya`. On x86-64 Linux and Windows that is pyke's CUDA build
(`+cuda13,tensorrt,nvrtx`), and the GPU pack ships its two provider libraries next to NVIDIA's
CUDA 13 and cuDNN 9 libraries.

pyke compiles that build with `CMAKE_CUDA_ARCHITECTURES=75;80;90`. The provider has no SASS and no
PTX an RTX 50-series GPU (Blackwell, sm_120) can use, so every request fails with
`cudaErrorNoKernelImageForDevice` (#10, RTX 5090, Docker `:cuda` and native Linux). Since 0.6.1
the runner falls back to the CPU in that case, which hides the error but not the problem.
pyke added sm_120 to its build scripts on 2026-09-23, but no `ort` release carries it, and pyke
ships no CUDA 12 build at all, so drivers older than R580 (CUDA 13) cannot use the GPU pack either
(asked for on HN).

The user approved the real fix on 2026-09-26: GPU builds that include sm_120, and a CUDA 12 option.

## Options

1. **Wait for pyke.** No date; still no CUDA 12; still starts at Turing.
2. **Microsoft's official onnxruntime GPU release, loaded with `ort`'s `load-dynamic`.** Microsoft
   publishes `onnxruntime-{linux,win}-x64-gpu_cuda12-<v>` and `..._cuda13-<v>` with every release
   (1.28.1, 1.28.2, 1.29.x and 1.30.0 all have the four packages). The ONNX Runtime library comes
   with its providers, built together; `ort` loads it with `dlopen`/`LoadLibrary` instead of linking.
3. **Build ONNX Runtime in CI** with our own SM list. Full control, but a multi-hour CUDA build per
   release and a toolchain to maintain.
4. **NV TensorRT-RTX EP.** JIT-compiles for the installed RTX GPU; different numerics (TF32, fp16
   tactics) that must pass parity; no pre-RTX GPUs.

Swapping only the provider libraries (Microsoft's `libonnxruntime_providers_cuda.so` under pyke's
statically linked core) is not an option: the provider bridge is an internal interface between
libraries of one build, with no compatibility promise across builds.

### Evidence for option 2 (measured 2026-09-26 on choso-wsl)

Microsoft ONNX Runtime **1.28.2** (same minor as pyke's 1.28.0, so the same operators and C API;
`ort` rc.13 with `api-24` needs 1.24 or newer). SASS and PTX from
`cuobjdump --list-elf` / `--list-ptx` (cuobjdump 13.4.92) on `libonnxruntime_providers_cuda.so`:

| Package | Download | Provider library | SASS (cubins) | PTX | Built with |
|---|---|---|---|---|---|
| linux-x64-gpu_cuda12-1.28.2.tgz | 424 MB | 621 MB | sm_60, 70, 75, 80, 86, 90a, 120a | compute_120 | CUDA 12.8 (PTX ISA 8.7) |
| linux-x64-gpu_cuda13-1.28.2.tgz | 241 MB | 280 MB | sm_75, 80, 86, 89, 90a, 100a, 120a | compute_120 | CUDA 13.0 (PTX ISA 9.0) |
| win-x64-gpu_cuda12-1.28.2.zip | 454 MB | not measured | | | |
| win-x64-gpu_cuda13-1.28.2.zip | 366 MB | not measured | | | |
| pyke 1.28.0 cuda13 (today) | | | sm_75, 80, 90 | none usable on sm_120 | |

- sm_120a SASS runs on compute capability 12.0 exactly (RTX 5090, 5080, RTX PRO 6000 Blackwell).
  203 of the 235 fatbin entries have sm_120a SASS; all 235 carry compute_120 PTX, which the driver
  JIT-compiles on a 12.x GPU for the rest (once, then cached in `~/.nv/ComputeCache`).
- The CUDA 12 build keeps sm_60 and sm_70 SASS: Pascal (GTX 10xx, sm_61 runs sm_60 code) and Volta
  work with it. The CUDA 13 build starts at Turing, as CUDA 13 itself does. Maxwell (GTX 9xx,
  sm_52) is in neither.
- `libonnxruntime.so.1.28.2` needs glibc 2.27 (pyke's static build needs 2.38).
- The provider links `libcudart`, `libcublas`, `libcublasLt` (`.so.12` or `.so.13`) and
  `libcurand.so.10` (`readelf -d`); cuDNN 9 and cuFFT are not in that list, so they are loaded
  at run time.
- Driver floor: CUDA 13 build, R580 or newer (unchanged). CUDA 12 build, R525 or newer through
  CUDA minor-version compatibility (R570 or newer for Blackwell, which needs it anyway, and for PTX
  JIT of 12.8 PTX).

Parity on the RTX 4090 (sm_89), `examples/parity` built with `--features cuda-dynamic`, fp32
graphs, goldens in `convert/out/goldens-all`, tolerances unchanged:

| Pack | Model | Cases | Decisions agree | Prob diff max |
|---|---|---|---|---|
| Microsoft 1.28.2 cuda13 + CUDA 13.x/cuDNN 9 libs of the current pack | laya:en | 477 | 100.00% | 1.0e-4 |
| same | laya:multilingual | 477 | 100.00% | 6.5e-6 |
| Microsoft 1.28.2 cuda12 + CUDA 12.8 wheels, cuDNN 9.26 | laya:en | 477 | 100.00% | 1.0e-4 |
| same | laya:multilingual | 477 | 100.00% | 6.5e-6 |

For reference, the pyke build on the same GPU: 100%, max 2.2e-4.

## Decision

Option 2, with the smallest change to what ships:

- `bin/ollaya` stays exactly as it is: pyke's ONNX Runtime linked statically, used for every CPU
  runner and for installs without a GPU pack.
- A second build of the same executable, `ollaya-cuda-runner`, built with
  `--features ollaya-runner/cuda-dynamic` (`ort/load-dynamic`), ships in the base archive for
  linux-amd64 and windows-amd64 at `lib/ollaya/ollaya-cuda-runner[.exe]`. It is outside the GPU
  pack so that the pack stays third-party files only and its `FILES.sha256` still changes only
  when a library changes, which is what lets the installers keep an unchanged 1 GB pack.
- The GPU pack (`lib/ollaya/cuda_v13`) holds Microsoft's ONNX Runtime 1.28.2 CUDA 13 build,
  unmodified: `libonnxruntime.so.1` (`onnxruntime.dll`), `libonnxruntime_providers_shared.so` and
  `libonnxruntime_providers_cuda.so`, plus the NVIDIA libraries it has today.
- The daemon (`crates/ollaya-server/src/launch.rs`): when the pack holds `libonnxruntime.so.1`,
  runners start from `ollaya-cuda-runner` with `ORT_DYLIB_PATH` set to it. On Linux `argv[0]` and
  `LD_LIBRARY_PATH` stay as they are; on Windows the runner copy inside the pack is made from
  `ollaya-cuda-runner.exe`. A pack with ONNX Runtime but no runner is not used, so the static
  build never loads another build's providers.
- ONNX models on the CPU keep starting from `bin/ollaya` (pyke's ORT), with or without the pack:
  `OLLAYA_DEVICE=cpu` starts them there, and with `auto` the daemon starts the GPU runner and,
  if it fails to load, a CPU runner from `bin/ollaya`. The GPU runner treats `auto` as the first
  GPU and never falls back to the CPU inside its own process. CPU numbers are the same as before
  this change. GGUF models run on llama.cpp and keep the GPU runner, as before.
- **A CUDA 12 pack, from 0.7.3** (it was deferred from 0.7.2): `lib/ollaya/cuda_v12`, Microsoft's
  cuda12 build with CUDA 12.8 and cuDNN 9 for CUDA 12 from NVIDIA's wheels
  (`packaging/cuda12-requirements.txt`) and llama.cpp's CUDA 12.8 `libggml-cuda.so`, for drivers
  without CUDA 13 support. The installers pick one pack by the driver's CUDA version
  (`install.sh`: 13 or newer gets `cuda_v13`, 12.x `cuda_v12`; `install.ps1` the same, or R580/R527
  from WMI), the daemon looks for `cuda_v13` first, and the Docker image `:cuda12` carries it.
  On the RTX 4090 every parity check gives the CUDA 13 pack's numbers (docs/distribution.md,
  "The CUDA 12 pack").
- Microsoft's version moves with `ort`: when `ort` moves to ONNX Runtime 1.x, the GPU pack moves
  to Microsoft's 1.x release, and both CPU and CUDA parity run again (PROJECT_NOTES: ORT changes
  only in a dedicated change).

## Consequences

- RTX 50-series GPUs get native SASS for most kernels and PTX for the rest. Parity was measured
  on sm_89 only; the reporter of #10 confirmed on 2026-09-26 that 0.7.2 runs on their RTX 5090
  (sm_120).
- With the pack installed, GPU runners run Microsoft's 1.28.2 and CPU runners pyke's 1.28.0, as
  without it. A GPU that fails to load under `auto` costs one extra runner start before the CPU
  runner.
- The base archive grows by one executable on x86-64 Linux and Windows. The CUDA 13 pack's ORT
  part shrinks or grows with Microsoft's build (provider 280 MB); the CUDA 12 pack is larger
  (provider 621 MB, cuBLAS 12).
- `CUDA_FORCE_PTX_JIT=1` is no longer a useful check on the 4090: the only PTX in Microsoft's
  build is compute_120, which an sm_89 GPU cannot JIT, so it fails at `cublasCreate`. It was useful
  for pyke's build, where it reproduced #10's failure mode on sm_89.
- The Docker `:cuda` image gets the new runner and pack from the same staging; its driver
  requirement (R580) is unchanged. A `:cuda12` tag is possible with the CUDA 12 pack.
- The llama.cpp CUDA backend in the pack (`libggml-cuda.so`) is built against CUDA 13; a CUDA 12
  pack needs llama.cpp's CUDA 12 build too.
- If Microsoft's packages ever drop an architecture we need, option 3 is the fallback; the runner
  side (`cuda-dynamic`, the pack layout) stays the same.
