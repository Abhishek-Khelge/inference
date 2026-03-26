# LLM Profiling Baseline

This project gives you a runnable Jupyter notebook for profiling a causal language model with `torch.profiler` on CPU and, when available, on a CUDA GPU.

## What is included

- `llm_profiling_baseline.ipynb`: the notebook version of the lesson
- `scripts/setup_env.sh`: creates a Python 3.11 virtual environment with the right packages

## Important note about GPU profiling

Your current machine is an Apple Silicon MacBook Air, so it can run the notebook locally on CPU, but it cannot run the CUDA profiling section.

For the GPU section, open this same notebook in a GPU-backed Linux environment such as:

- Google Colab with a T4/L4/A100 runtime
- Runpod with a Jupyter template
- Paperspace Gradient
- A Linux VM with NVIDIA drivers and CUDA

An LLM inference API is not enough for `torch.profiler`. The notebook needs direct access to the Python process and the GPU device.

## Local setup on this machine

```bash
./scripts/setup_env.sh mac
source .venv/bin/activate
jupyter lab
```

Then open:

- `llm_profiling_baseline.ipynb`

## CUDA setup in a GPU notebook or Linux VM

```bash
./scripts/setup_env.sh cuda
source .venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

## What the notebook measures

- End-to-end latency
- Tokens per second
- Model memory footprint
- CUDA peak memory, when available
- Key operator timings from `torch.profiler`

## Default model

The notebook uses `gpt2-medium` by default to match your lesson. If that is too slow or memory-heavy, change:

```python
model_name = "gpt2-medium"
```

to:

```python
model_name = "distilgpt2"
```
