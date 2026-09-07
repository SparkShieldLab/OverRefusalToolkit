# Reported results and reproducibility / 实验结果与复现说明

## Source / 来源

The Spark 13B tables in the READMEs and announcement reproduce the project-provided experiment summary. The supplied release materials contain aggregate values, but no per-example generations, WildGuard scoring logs, or complete run manifests. These numbers were checked for consistency and arithmetic; they were not independently reproduced during release preparation.

中英文 README 与宣传稿中的星火 13B 数据来自项目提供的实验汇总。本次材料未附逐条生成结果、WildGuard 评分日志及完整运行 manifest。整理时核对了表间一致性与数值计算，没有重新运行或独立复现这些模型实验。

## Metrics / 指标

The implementation in [`compute_metrics`](../src/prosafeprune/evaluate.py) uses:

- **Compliance / 可用性**: the unweighted mean of `100 − Response refusal (%)` across ORBench, PHTest, XSTest_safe and OKTest.
- **Safety / 安全性**: the unweighted mean of `100 − Harmful response (%)` across AdvBench and JailbreakBench.
- **Overall / 综合得分**: `(Compliance + Safety) / 2`.

All scores use a 0–100 scale. The displayed overall values are rounded to three decimal places. Changes in percentage-based metrics are percentage-point differences, not relative percentage improvements. Dataset means are weighted equally within each group, regardless of dataset size.

“可用性”在这里是非拒绝率指标，并非正确率或回答质量评分。“安全性”是评估器判定的非有害响应率，并非对未知攻击的安全保证。

## Known configuration / 已知配置

The supplied summary identifies the model as Spark 13B. All displayed pruned configurations use `7m`: q/k/v/o and gate/up/down projections. Layer indices are zero-based, matching the implementation.

| Experiment | Layers | Rank | λ | Modules | Decoder |
|---|---|---:|---:|---|---|
| Pruning A | 14 | 9 | 1.0 | 7m | Greedy |
| Pruning B | 18, 21, 23 | 4 | 0.6 | 7m | Greedy |
| Pruning C | 18, 21 | 4 | 1.0 | 7m | Greedy |

## Information needed for exact reproduction / 精确复现仍需的信息

- Exact Spark 13B checkpoint and tokenizer revision, including chat template and thinking-mode settings.
- Per-dataset evaluation sample counts, selected example IDs and random seed.
- Activation selection, sample counts and generated basis manifests.
- Generation limits, dtype, library versions and hardware.
- WildGuard checkpoint revision, raw judgments and evaluation manifests.

仓库示例的 `--sample-count 50`、默认种子 `42` 及生成上限 `128` 是工具默认值，不能自动视为上述实验的实际参数。

To record a new experiment, run both baseline and candidate models with the same evaluation subset and settings, retain the generated manifests and scoring artifacts, and use `python -m prosafeprune.summarize_evaluations` to collect the results. Consult the [English](../README.md) or [Chinese](../README_zh.md) README for the workflow.
