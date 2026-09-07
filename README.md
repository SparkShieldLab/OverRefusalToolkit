<p align="center">
  <a href="./README_zh.md">简体中文</a> |
  <strong>English</strong>
</p>

# OverRefusalToolkit

OverRefusalToolkit is an LLM over-refusal analysis and mitigation toolkit based on [ProSafePrune](https://openreview.net/forum?id=QkHKaPfRAB). It provides activation-subspace analysis, projection pruning, and a unified evaluation pipeline. The implementation is refactored and extended from research code provided by the original authors.

## Features

- **ProSafePrune:** identifies directions related to over-refusal from safe, unsafe, and pseudo-harmful activations, then applies projection pruning to selected layers and modules;
- **Evaluation:** compares responses to benign requests and safety on harmful requests using fixed evaluation subsets;
- **Model support:** works with standard Hugging Face causal language models and shared-attention projections in MoE models.

## Results

The following results demonstrate the performance of the toolkit on Spark 13B. Compliance is the average score across ORBench, PHTest, XSTest-safe, and OKTest. Safety is the average score across AdvBench and JailbreakBench. The overall score is the arithmetic mean of compliance and safety. Higher values are better for all metrics.

| Configuration | Compliance | Safety | Overall |
|---|---:|---:|---:|
| Baseline | 66.375 | 100.000 | 83.188 |
| `L14_R9_Lam1_M7m` | 75.625 | 100.000 | 87.813 |
| `L18_21_23_R4_Lam0.6_M7m` | 84.000 | 99.500 | 91.750 |
| `L18_21_R4_Lam1_M7m` | 92.125 | 98.000 | 95.063 |

Different pruning parameters produce operating points with different compliance–safety trade-offs. Among the configurations shown above, compliance increased by up to 25.75 percentage points, with a corresponding 2.00-point decrease in safety.

These are representative results on the fixed evaluation subsets included with the repository. Results depend on the target model, pruning parameters, and decoding configuration.

## Installation

```bash
git clone https://github.com/SparkShieldLab/OverRefusalToolkit.git
cd OverRefusalToolkit
conda activate YOUR_ENV
python -m pip install -e .
```

## Configuration

Set the required paths in `scripts/runtime_config.sh`:

```bash
export PROSAFEPRUNE_MODEL_ID="MODEL_ID"
export PROSAFEPRUNE_MODEL_PATH="/path/to/model"
```

Load the configuration before running commands manually:

```bash
source scripts/runtime_config.sh
```

Activation collection, analysis, and pruning do not require WildGuard. Configure `PROSAFEPRUNE_WILDGUARD_PATH` only when using the provided evaluation pipeline.

To collect attention projections only, enable the following setting:

```bash
export PROSAFEPRUNE_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
```

This setting controls activation collection; use the shorthand `--modules att` during pruning.

For models whose chat templates support the `enable_thinking` argument, set `PROSAFEPRUNE_DISABLE_THINKING=true` to render prompts with `enable_thinking=False`.

## Models

Base models, pruned models, and WildGuard weights are not distributed. Prepare a Hugging Face-compatible base model locally; pruned models are produced by the pipeline. WildGuard is additionally required for evaluation.

## Data

The repository includes fixed JSON subsets for activation construction and evaluation under `data/`; see [`data/README.md`](data/README.md).

## ProSafePrune

```bash
python -m prosafeprune.collect_activations --batch-size 1
python -m prosafeprune.analyze_layers --window 5
python -m prosafeprune.analyze_rank
python -m prosafeprune.visualize
python -m prosafeprune.build_basis
python -m prosafeprune.prune \
  --layers 15-17 --rank 16 --lam 0.9 --modules att
```

Pruning requires explicit layers, rank, and lambda. Module groups are `att` (q/k/v/o), `mlp` (gate/up/down), and `7m` (all seven projections). `7m` means seven modules, not a model size.

## Evaluation

```bash
python -m prosafeprune.evaluate full \
  --model-path /path/to/model --sample-count 50 --batch-size 4

# Summary
python -m prosafeprune.summarize_evaluations
```

## MoE models

Current MoE support is limited to shared attention:

```bash
export PROSAFEPRUNE_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
```

Use `--modules att`. Routers, shared experts, routed experts, and packed expert parameters are not supported.

## Outputs

```text
artifacts/models/{MODEL_ID}/
├── activations/
├── analysis/
├── subspaces/
├── pruned/
├── evaluations/greedy/
└── evaluation_tables/
```

## References

- [ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs](https://openreview.net/forum?id=QkHKaPfRAB)

Please cite the original ProSafePrune paper when using this project.

## Acknowledgements

We sincerely thank the Anhui Laboratory for Safe Artificial Intelligence in the Yangtze River Delta for supporting this project.

## About the Anhui Laboratory for Safe Artificial Intelligence in the Yangtze River Delta

The Anhui Laboratory for Safe Artificial Intelligence in the Yangtze River Delta is dedicated to advancing trustworthy and safe AI. Our work spans policy-sensitive scenarios, model content safety, agent safety, and rigorous safety evaluation, with a particular focus on safety challenges under complex real-world demands. We welcome research and industry collaborations in these areas.

<p>
  <a href="https://sai.xingdun-ai.com/home"><img src="https://img.shields.io/badge/Website-Official_Site-1677FF?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Official Website" /></a>
  <a href="https://open.weixin.qq.com/qr/code?username=gh_89d544e1b8aa"><img src="https://img.shields.io/badge/WeChat-Follow_Us-07C160?style=flat-square&amp;logo=wechat&amp;logoColor=white" alt="WeChat" /></a>
</p>

<p align="center">
  <strong>Scan the QR code below to join our WeChat group.</strong>
</p>
<p align="center">
  <img src="assets/wechat-group-qr.png" alt="WeChat group QR code" width="220" />
</p>

## License

The project code is released under the [MIT License](LICENSE). Data under `data/` remain subject to the terms of their respective sources.
