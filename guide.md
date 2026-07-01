# ERNIE Image Turbo with Nunchaku Lite — Setup & Inference Guide

This guide records the exact, verified command sequence used to get ERNIE Image
Turbo running with `nunchaku_lite` on an **NVIDIA RTX 5090 (Blackwell, sm120a)**
using **CUDA 12.8** and a **uv-managed Python 3.12** virtual environment.

It covers both the **INT4** and **FP4 (NVFP4)** inference paths.

---

## Environment Summary

| Component         | Value                                      |
| ----------------- | ------------------------------------------ |
| OS                | Linux                                      |
| GPU               | NVIDIA GeForce RTX 5090 (compute 12.0)     |
| CUDA toolkit      | 12.8 (`/usr/local/cuda-12.8`)              |
| Python            | 3.12.3                                     |
| PyTorch           | 2.11.0+cu128                               |
| nunchaku_lite     | 0.2.0dev                                   |
| nunchaku_lite_kernels | built from `./nunchaku-lite-kernels`  |
| uv                | 0.11.18                                    |

> **Why CUDA 12.8?** Blackwell `sm120a` requires nvcc ≥ 12.8. The INT4 BF16
> kernel template and the NVFP4 kernel are both compiled for `sm120a`.
> Hopper (sm90) is **not supported** by this codebase — see
> `nunchaku-lite-kernels/setup.py::get_sm_targets()`.

---

## Timeline of Working Commands

All commands are run from the repository root:

```bash
cd /home/jyao/ADEO/DiT/nunchaku-lite
```

### Step 1 — Create the virtual environment

```bash
uv venv --python 3.12 .venv
```

Creates `.venv/` with Python 3.12.3. (Python 3.10–3.13 are all supported per
`pyproject.toml`; 3.12 is the recommended mainstream choice.)

### Step 2 — Activate the virtual environment

```bash
source .venv/bin/activate
```

The shell prompt should now show `(.venv)`.

### Step 3 — Install PyTorch with CUDA 12.8

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Installs `torch==2.11.0+cu128`. **This must match the system nvcc (12.8)** or
the native CUDA extension build in Step 4 will fail.

### Step 4 — Build and install the native kernels package

```bash
uv pip install ./nunchaku-lite-kernels
```

Compiles the `nunchaku_lite_kernels._C` CUDA extension for `sm120a` (auto-
detected via `NUNCHAKU_INSTALL_MODE=FAST`). This step takes several minutes.

If the build complains about a missing `ninja`, run this first and retry:

```bash
uv pip install ninja
```

### Step 5 — Install the main nunchaku_lite package

```bash
uv pip install .
```

Installs the Python runtime plus dependencies (`diffusers`, `transformers`,
`accelerate`, `safetensors`, `huggingface-hub`, `peft`).

### Step 6 — Verify the native kernels loaded

```bash
python -c "from nunchaku_lite.ops.backend import get_ops; print(get_ops())"
```

A successful run prints the native ops backend object. If you see an
`ImportError` or a Hugging Face fallback message, the kernel build in Step 4
did not succeed.

### Step 7 — Verify precision auto-detection

```bash
python -c "import torch; from nunchaku_lite.utils import get_precision; print(get_precision('auto', 'cuda'))"
```

On the RTX 5090 this prints `fp4` (because sm ≥ 100). This is why INT4 runs
must pass `precision="int4"` explicitly — `"auto"` selects FP4 on Blackwell.

### Step 8 — Run INT4 inference

```bash
python -c "
import torch
from diffusers import ErnieImagePipeline
from nunchaku_lite import load_nunchaku_pipeline

pipe = load_nunchaku_pipeline(
    'Baidu/ERNIE-Image-Turbo',
    pipeline_cls=ErnieImagePipeline,
    checkpoint='rootonchair/ERNIE-Image-Turbo-nunchaku-lite/svdq-int4_r32-ernie-image-turbo-zero-svdq-fix-bias.safetensors',
    precision='int4',
    torch_dtype=torch.bfloat16,
    device='cuda',
).to('cuda')

image = pipe(
    prompt='Astronaut in a jungle, cold color palette, muted colors, detailed, 8k',
    height=1264,
    width=848,
    num_inference_steps=8,
    guidance_scale=1.0,
    use_pe=True,
).images[0]

image.save('output.png')
print('saved output.png')
"
```

**Result:** Exit code 0. Saved `output.png` (1,994,308 bytes).

> **Checkpoint note:** The original INT4 checkpoint
> `svdq-int4_r32-ernie-image-turbo.safetensors` is **not present** on the Hub
> (HTTP 404). Use the fixed-bias variant
> `svdq-int4_r32-ernie-image-turbo-zero-svdq-fix-bias.safetensors`, which
> zeroes the exported SVDQ target biases to avoid a known activation-shift
> folded-bias issue that can produce noise-like outputs.

### Step 9 — Run FP4 (NVFP4) inference

```bash
python -c "
import torch
from diffusers import ErnieImagePipeline
from nunchaku_lite import load_nunchaku_pipeline

pipe = load_nunchaku_pipeline(
    'Baidu/ERNIE-Image-Turbo',
    pipeline_cls=ErnieImagePipeline,
    checkpoint='rootonchair/ERNIE-Image-Turbo-nunchaku-lite/svdq-nvfp4_r32-ernie-image-turbo.safetensors',
    precision='fp4',
    torch_dtype=torch.bfloat16,
    device='cuda',
).to('cuda')

image = pipe(
    prompt='Astronaut in a jungle, cold color palette, muted colors, detailed, 8k',
    height=1264,
    width=848,
    num_inference_steps=8,
    guidance_scale=1.0,
    use_pe=True,
).images[0]

image.save('output_fp4.png')
print('saved output_fp4.png')
"
```

**Result:** Recommended path on Blackwell. FP4 is the native fast kernel on
sm120a — no `UserWarning`, and the model card recommends it for both speed
and quality on Blackwell GPUs.

---

## Available Checkpoints

Only two checkpoints exist in `rootonchair/ERNIE-Image-Turbo-nunchaku-lite`:

| File                                                                       | Precision | Notes                                              |
| -------------------------------------------------------------------------- | --------- | -------------------------------------------------- |
| `svdq-int4_r32-ernie-image-turbo-zero-svdq-fix-bias.safetensors`           | INT4      | Fixed-bias variant; avoids folded-bias noise.      |
| `svdq-nvfp4_r32-ernie-image-turbo.safetensors`                             | NVFP4     | Recommended on Blackwell for speed and quality.    |

---

## How ERNIE Image Turbo Is Supported

There is **no ERNIE-specific adapter** in this repo. The built-in adapters are
`flux`, `flux2`, `qwen_image`, `sdxl`, `z_image`, and a generic `manifest`
adapter. ERNIE Image Turbo works through the **generic manifest adapter**,
auto-selected from the checkpoint's `quantization_config.runtime_manifest`
metadata when `target="auto"` (the default).

### INT4 kernel call chain

```
SVDQW4A4Linear.forward()                 # src/nunchaku_lite/linear.py
  └─ fp4 = (self.precision == "nvfp4")   # → False for int4
  └─ svdq_gemm_w4a4_cuda(fp4=False)      # src/nunchaku_lite/ops/gemm.py
       └─ ops.gemm_w4a4(fp4=False)       # pybind → C++
            └─ nunchaku::kernels::gemm_w4a4()        # kernels/zgemm/gemm_w4a4.cu
                 └─ invoke_launch(BF16, use_fp4=false)
                      └─ GEMMConfig_W4A4_BF16, USE_FP4=false
                           └─ gemm_w4a4_kernel<Epilogue, ACT_UNSIGNED>
                                └─ mma.sync ... s32 (INT4 tensor cores)
```

### Key source files

| File                                                                          | Role                                                    |
| ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| `nunchaku-lite-kernels/.../kernels/zgemm/gemm_w4a4.cu`                        | Top-level `gemm_w4a4()` dispatcher                      |
| `nunchaku-lite-kernels/.../kernels/zgemm/gemm_w4a4_launch_impl.cuh`           | Kernel launch; `if constexpr (!USE_FP4)` INT4 branch    |
| `nunchaku-lite-kernels/.../kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu`       | Explicit INT4 BF16 template instantiation               |
| `nunchaku-lite-kernels/.../kernels/zgemm/mma.cuh`                             | PTX `mma.sync` tensor-core instructions                 |
| `src/nunchaku_lite/linear.py`                                                 | `SVDQW4A4Linear` — sets `fp4` from precision            |
| `src/nunchaku_lite/core.py`                                                   | `load_nunchaku_pipeline`, precision validation          |
| `src/nunchaku_lite/ops/backend.py`                                            | Loads local `nunchaku_lite_kernels` then HF fallback    |

### Precision → buffer layout

|                  | INT4               | NVFP4             |
| ---------------- | ------------------ | ----------------- |
| `group_size`     | 64                 | 16                |
| `wscales` dtype  | `bfloat16`         | `float8_e4m3fn`   |
| `wcscales`       | `None`             | present           |
| Kernel scales    | int group scales   | fp8 scales + alpha|

---

## Hardware Compatibility Notes

| Architecture        | SM     | INT4  | FP4     | Status                  |
| ------------------- | ------ | ----- | ------- | ----------------------- |
| Turing (RTX 20)     | 75     | ✅    | ❌      | INT4 only               |
| Ampere (A100)       | 80     | ✅    | ❌      | INT4 only               |
| Ampere (RTX 30)     | 86     | ✅    | ❌      | INT4 only               |
| Ada (RTX 40)        | 89     | ✅    | ❌      | INT4 only               |
| Hopper (H100)       | 90     | ❌    | ❌      | **Not supported**       |
| Blackwell (RTX 50)  | 120a   | ✅\*  | ✅      | FP4 recommended         |

\* INT4 works on Blackwell but emits a `UserWarning` that FP4 is faster.

Hopper (sm90) is explicitly rejected by both `get_sm_targets()` at build time
and `_validate_quantization_compatibility()` at runtime. Supporting Hopper
would require WGMMA + TMA kernels, which are not implemented.

---

## Troubleshooting

### `404 Not Found` on checkpoint download

The original `svdq-int4_r32-ernie-image-turbo.safetensors` was never uploaded.
Use `svdq-int4_r32-ernie-image-turbo-zero-svdq-fix-bias.safetensors` instead.

### `Unsupported SM 90`

You are on a Hopper GPU. Nunchaku Lite does not support Hopper. Use the full
`nunchaku` package or run BF16/FP8 diffusers inference directly.

### INT4 produces noise-like output

Ensure you are using the **fixed-bias** INT4 checkpoint
(`...-zero-svdq-fix-bias.safetensors`), not the original export. On Blackwell,
prefer the FP4 checkpoint for both speed and quality.

### Native ops fallback to Hugging Face kernels

If `python -c "from nunchaku_lite.ops.backend import get_ops; print(get_ops())"`
raises an `ImportError` about Hugging Face kernels, the Step 4 kernel build
failed. Re-run `uv pip install ./nunchaku-lite-kernels` and check that `nvcc`
is on `PATH` and matches the torch CUDA build (cu128).

### nvcc not found during build

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

Then re-run Step 4.
