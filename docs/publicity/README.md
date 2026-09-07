# 宣传材料

## 公众号长文

[该拒绝的拒绝，该回答的回答：ICLR 2026 ProSafePrune 让大模型少些“误拒”](announcement_zh.md)

长文以长三角安全人工智能安徽省实验室、合肥工业大学、科大讯飞研究院合作开展的 ProSafePrune 研究为主题，串联方法介绍、论文结果、星火 13B 剪枝实验与开源工具实践；两张配图已按正文顺序放入 `images/`，可随 Markdown 一起下载。

## 发布标题

该拒绝的拒绝，该回答的回答：ICLR 2026 ProSafePrune 让大模型少些“误拒”

## 公众号摘要

长三角安全人工智能安徽省实验室、合肥工业大学、科大讯飞研究院合作开展 ProSafePrune 研究，成果发表于 ICLR 2026。本文介绍如何通过投影剪枝缓解大模型“误拒”，以及星火 13B 验证与开源实践。

## 社群发布短文

长三角安全人工智能安徽省实验室、合肥工业大学、科大讯飞研究院合作完成的 ProSafePrune 研究发表于 ICLR 2026。该方法从模型内部表示出发，通过低秩投影剪枝缓解大模型对合理请求的过度拒绝。围绕这项合作研究，我们进一步开展星火 13B 模型验证，并开源 OverRefusalToolkit，提供激活分析、剪枝配置、生成和统一评估流程。

在固定评估子集上，星火 13B 的一组代表性剪枝配置将可用性由 66.375 提升至 92.125，安全性由 100.000 变为 98.000。工具提供可调节的剪枝参数与统一评估流程，帮助比较不同配置的效果。欢迎交流模型适配、评估经验与改进建议。

项目地址：[SparkShieldLab/OverRefusalToolkit](https://github.com/SparkShieldLab/OverRefusalToolkit)

## 英文仓库描述

A toolkit for studying and mitigating LLM over-refusal with ProSafePrune projection pruning. Provides activation analysis, layer and rank selection support, model pruning, and evaluation of benign-request compliance and harmful-response safety for Hugging Face causal language models.

## 配图与排版

| 图片 | 放置位置 | 用途 |
|---|---|---|
| [01_overrefusal_intent_boundary.png](images/01_overrefusal_intent_boundary.png) | 场景与指标介绍 | 解释敏感概念与合理请求的关系 |
| [02_prosafeprune_projection_pruning.png](images/02_prosafeprune_projection_pruning.png) | ProSafePrune 方法介绍 | 展示激活、子空间与权重更新 |

沿用过度拒绝场景和 ProSafePrune 方法两张示意图，正文图注补充了符号及实现细节。图 2 中基向量与投影矩阵的区别，以正文公式和代码为准。发布时保留图注。图 2 较宽，适合支持点击查看大图的排版。

公众号编辑器通常不能直接使用仓库相对图片路径；应上传对应图片，并将相对文档链接换成 GitHub 页面地址。公式需在排版工具中渲染，避免直接显示 LaTeX 源码。

## 结果口径

原论文结果与本项目星火 13B 实验已分开标注。百分比指标之间的差值使用“百分点”；综合得分差值使用“分”。发布时同时保留可用性与安全性的变化，结果来源及复现信息见[结果说明](../results.md)。
