<p align="center">
  <strong>简体中文</strong> |
  <a href="./README.md">English</a>
</p>

# OverRefusalToolkit

OverRefusalToolkit 是基于 [ProSafePrune](https://openreview.net/forum?id=QkHKaPfRAB) 构建的大语言模型过度拒绝分析与缓解工具，覆盖激活子空间分析、投影剪枝和统一评估流程。本仓库中的实现基于原作者提供的研究代码进行了重构与工程化改进。

## 主要功能

- **ProSafePrune**：利用安全、有害和伪有害样本的激活表示定位与过度拒绝相关的方向，并对指定层和模块进行投影剪枝；
- **统一评估**：在固定评估子集上比较模型对正常请求的回答能力与对有害请求的安全性；
- **模型支持**：适用于标准 Hugging Face Causal LM，并支持对 MoE 模型的 shared attention 进行分析和剪枝。

## 实验结果

以下展示星火 13B 上的实验效果。可用性为 ORBench、PHTest、XSTest-safe 和 OKTest 的平均得分，安全性为 AdvBench 和 JailbreakBench 的平均得分，综合得分为可用性与安全性的算术平均。所有指标均为越高越好。

| 配置 | 可用性 | 安全性 | 综合得分 |
|---|---:|---:|---:|
| Baseline | 66.375 | 100.000 | 83.188 |
| `L14_R9_Lam1_M7m` | 75.625 | 100.000 | 87.813 |
| `L18_21_23_R4_Lam0.6_M7m` | 84.000 | 99.500 | 91.750 |
| `L18_21_R4_Lam1_M7m` | 92.125 | 98.000 | 95.063 |

不同剪枝参数形成了具有不同可用性—安全性侧重的运行配置。在以上结果中，可用性最高提高 25.75 个百分点，对应的安全性下降 2.00 个百分点。

以上为仓库固定评估子集上的代表性实验结果，用于展示不同配置下的可用性—安全性变化；具体结果依赖模型、剪枝参数和解码配置。

## 安装

```bash
git clone https://github.com/SparkShieldLab/OverRefusalToolkit.git
cd OverRefusalToolkit
conda activate YOUR_ENV
python -m pip install -e .
```

## 配置

修改 `scripts/runtime_config.sh` 中的必填项：

```bash
export PROSAFEPRUNE_MODEL_ID="MODEL_ID"
export PROSAFEPRUNE_MODEL_PATH="/path/to/model"
```

手动执行命令前加载配置：

```bash
source scripts/runtime_config.sh
```

激活收集、分析和剪枝不依赖 WildGuard。只有执行本项目提供的评估流程时，才需要配置 `PROSAFEPRUNE_WILDGUARD_PATH`。

如只收集 attention projection，在配置中启用：

```bash
export PROSAFEPRUNE_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
```

该配置控制激活收集范围；剪枝时使用简写 `--modules att`。对于 chat template 支持 `enable_thinking` 的模型，可以设置 `PROSAFEPRUNE_DISABLE_THINKING=true`。

## 模型准备

本仓库不提供基础模型、剪枝模型或 WildGuard 权重。使用者需要自行准备兼容 Hugging Face 格式的基础模型；剪枝模型由项目运行后生成。执行评估时还需要提供 WildGuard 的本地模型路径。

## 数据

仓库在 `data/` 中提供用于激活构建和评估的固定 JSON 子集，文件说明见 [`data/README.md`](data/README.md)。

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

剪枝必须显式选择 layers、rank 和 lambda。模块组包括 `att`（q/k/v/o）、`mlp`（gate/up/down）和 `7m`（全部七个 projection）；`7m` 不是模型规模。

## 评估

```bash
python -m prosafeprune.evaluate full \
  --model-path /path/to/model --sample-count 50 --batch-size 4

# 汇总
python -m prosafeprune.summarize_evaluations
```

## MoE 模型

当前 MoE 只支持 shared attention。激活收集使用：

```bash
export PROSAFEPRUNE_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
```

剪枝直接使用 `--modules att`，无需分别指定 q/k/v/o。Router、shared expert、routed expert 和 packed expert 参数不在支持范围。

## 输出

```text
artifacts/models/{MODEL_ID}/
├── activations/
├── analysis/
├── subspaces/
├── pruned/
├── evaluations/greedy/
└── evaluation_tables/
```

## 参考工作

- [ProSafePrune: Projected Safety Pruning for Mitigating Over-Refusal in LLMs](https://openreview.net/forum?id=QkHKaPfRAB)

使用本项目时，请引用 ProSafePrune 原始论文。

## 致谢

衷心感谢长三角安全人工智能安徽省实验室的支持。

## 关于长三角安全人工智能安徽省实验室

长三角安全人工智能安徽省实验室致力于推动可信与安全人工智能的发展。我们的工作涵盖政策敏感场景、模型内容安全、智能体安全和严格的安全评估，并尤其关注复杂现实需求下的安全问题。我们欢迎这些方向的科研合作与产业合作，欢迎通过以下渠道与我们联系。

<p>
  <a href="https://sai.xingdun-ai.com/home"><img src="https://img.shields.io/badge/Website-Official_Site-1677FF?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="官方网站" /></a>
  <a href="https://open.weixin.qq.com/qr/code?username=gh_89d544e1b8aa"><img src="https://img.shields.io/badge/%E5%BE%AE%E4%BF%A1-WeChat-07C160?style=flat-square&amp;logo=wechat&amp;logoColor=white" alt="微信公众号" /></a>
</p>

<p align="center">
  <strong>扫描下方二维码加入微信群。</strong>
</p>
<p align="center">
  <img src="assets/wechat-group-qr.png" alt="微信群二维码" width="220" />
</p>

## 许可证

项目代码采用 [MIT License](LICENSE)。`data/` 中的数据仍遵循其各自来源的许可条款。
