# Workload Adaptation Evaluation

Streams real scientific data from [The Well](https://polymathic-ai.org/the_well/) (Post Neutron Star Merger dataset), compresses each timestep with GPUCompress's NN-based ALGO_AUTO, and plots how the neural network adapts to novel workloads via online reinforcement.

## Prerequisites

```bash
# Activate the Python virtual environment
source venv/bin/activate

# Install dependencies (one-time)
pip install huggingface_hub aiohttp requests

# Build the library
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && cd ..
```

## Run the Evaluation

```bash
source venv/bin/activate
export LD_LIBRARY_PATH=/tmp/lib:$(pwd)/build:$LD_LIBRARY_PATH
python eval/workload_adaptation.py
```

This will:
1. Stream 5 fields (density, internal_energy, temperature, pressure, entropy) from HuggingFace — no full 13 GB download needed
2. Compress 40 timesteps per field (200 total steps) with ALGO_AUTO + active learning + reinforcement
3. Save results to `eval/workload_adaptation.csv`
4. Plot MAPE + reinforcement rate to `eval/workload_adaptation.png`

## Adjusting Parameters

Edit the configuration block at the top of `eval/workload_adaptation.py`:

```python
FIELDS = ["density", "internal_energy", "temperature", "pressure", "entropy"]
N_TIMESTEPS = 40          # timesteps per field (max 181)
ERROR_BOUND = 0.001       # lossy quantization error bound
EXPLORATION_THRESH = 0.15 # exploration triggers when ratio MAPE > this
REINFORCE_LR = 0.05       # SGD learning rate for online reinforcement
REINFORCE_MAPE = 0.20     # MAPE threshold passed to set_reinforcement
```

### Learning rate (`REINFORCE_LR`)

Controls how aggressively the NN updates its weights after each compression.

| Value  | Behavior |
|--------|----------|
| 0.0001 | Very slow adaptation — predictions barely change between steps |
| 0.001  | Gradual learning — takes ~100+ steps to converge on new data |
| 0.01   | Moderate — converges within ~30-50 steps per workload |
| **0.05** | **Default** — fast adaptation, good for OOD data like this dataset |
| 0.1    | Aggressive — may overshoot and oscillate between steps |

Higher values help when the data is very different from the NN's training set (out-of-distribution). Lower values are better when the NN already makes reasonable predictions and you want stable fine-tuning.

### Other knobs

- **`N_TIMESTEPS`**: More timesteps gives the NN more time to adapt per field. Set to 181 for the full simulation.
- **`ERROR_BOUND`**: Set to `0.0` for lossless compression, or increase (e.g. `0.01`) for higher compression ratios with lossy quantization.
- **`EXPLORATION_THRESH`**: Lower values (e.g. `0.05`) trigger more exploration of alternative algorithms. Higher values (e.g. `0.30`) explore less.
- **`FIELDS`**: Any subset of `["density", "internal_energy", "temperature", "pressure", "entropy", "electron_fraction"]`.

## Output

- `eval/workload_adaptation.csv` — per-step results with predicted vs actual metrics
- `eval/workload_adaptation.png` — dual-axis MAPE + reinforcement rate plot
- `eval/experience_workload.csv` — raw experience samples collected during active learning
