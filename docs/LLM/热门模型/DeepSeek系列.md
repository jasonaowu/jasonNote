---
article: false
title: DeepSeek系列
order: 2
---

# 问题和结论

1. MOE 能否做知识蒸馏？ <span style="color:blue; background-color:lightblue;">**可以**</span>
2. MOE 相比 Dense 的优势？ <span style="color:blue; background-color:lightblue;">**节约计算成本**</span>
3. 什么结构能用 MOE？ **任何 FFN** → 多任务问题
4. 在已知 MOE 有负载不均衡问题的前提下，为啥目前大模型都开始抛弃传统 Transformer 架构，转投 MOE？ **便宜**
5. 个人原来的理解：MOE 只能节约训练和推理的计算量，不能节约存储量；模型蒸馏可以节约计算量，也可以节约存储量，是否正确？ **正确**

> **错误观点**：
>
> 1. MOE 是为了减小网络结构？ ❌，相反，MOE的初衷是为了在保证较低计算量的同时，增加模型参数，使模型更强
>
> 原来以为MOE是针对深层网络做的优化，将深层网络变为浅层网络，但是实际是将中间层参数数量从 N，降低为 N/E，分散到E个专家上

## MoE 模型基本特性

与稠密模型相比，对于给定的计算预算，MoE 模型提供更高效的训练。这是因为门控网络仅将 token 发送到一部分专家，从而减少了计算负载。因此，模型的容量（其参数总数）可以增加，而不会成比例地增加计算需求。在推理期间，仅使用部分专家，因此 MoE 能够执行比稠密模型更快的推理。但是，整个模型需要加载到内存中，而仅仅是正在使用的专家。

MoE 中实现更高计算效率的稀疏性来自于这样一个事实：特定的 token 只会被路由到一部分专家。专家的数量以及如何选择专家取决于门控网络的实现，但一种常见的方法是 top k。门控网络首先预测每个专家的概率值，然后将 token 路由到 top k 个专家以获得输出。但是，如果所有 token 始终都发送到相同的专家子集，则训练效率会降低，而其他专家最终会训练不足。为了缓解这个问题，引入了负载均衡损失，以鼓励均匀路由到所有专家。

专家的数量和选择 top k 个专家是设计 MoE 的重要因素。更多的专家数量允许扩展到更大的模型，而不会增加计算成本。这意味着模型具有更高的学习能力，但是，超过某个点后，性能增益往往会减少。选择的专家数量需要与服务模型的推理成本相平衡，因为整个模型都需要加载到内存中。同样，在选择 top k 时，训练期间较低的 top k 会导致较小的矩阵乘法，如果通信成本足够大，则会浪费计算资源。但是，在推理期间，较高的 top k 通常会导致较慢的推理速度。

## 参考资料

- [MOE 介绍](https://kevincheung2259.github.io/2024/09/13/MOE-Intro/index.html)
- [DeepSeek 技术解析](https://deepseek.csdn.net/67fa2941da5d787fd5cb6acb.html)

## 研究问题

以 Transformer 原文中 FFN 部分的参数量进行计算，如果换成 MOE 架构，参数量是多少，为啥能够节约计算时间？

## DeepSeek 研究脉络

💡 **从稠密模型到混合专家，再到推理方向**

回顾 DeepSeek 过去一年多发表的核心论文，我们大致能将其研究分为两条主要脉络：　

- **基座模型（Foundation Models）**：从最早的 Dense（稠密）结构一路演进到 MOE（混合专家）模式，并在这个过程中不断发明和采用新的高效训练算法。
- **推理能力（Reasoning）**：包括解数学题、代码生成、逻辑问答乃至定理证明等，更强调大模型的"思考深度"，并在如何进行强化学习方面进行了连续多次创新。

在阅读这份逐篇解读之前，可以先记住 DeepSeek 的几大特色：对实验和数据极度重视、有足够的冒险精神尝试新架构和新算法、且真正愿意分享内部研究细节，为社区提供可复现的技术报告。　

# MOE 基本原理

MOE 全称是 Mixture of Experts，也就是混合专家模型

## 最最最原始版

### 组成

1. **稀疏 MOE 层**：n 个专家 FFN
2. **路由**：token 到 top-K 个专家

![MOE 基本结构](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NWVjNjk1ZmU5YmUyNDA3ZGZlZGQ3MzljZWU5NzVhZWJfQ1pESDhYRFRIM1QwNVEwTzRJSVNCdWZOS1FkNHJ5Q2JfVG9rZW46UnA1M2Jja25Cb2c2MlJ4TTc5SGNtMFE1bmdlXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)
![MOE 结构图2](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NzMzNzMyZWE5ZDFkYjc1MTUwODE0OTViZjI4MWFjNGNfdTRrMVc0UXJnemtRekFWNTNhYzZwMXRBc2plak9naWZfVG9rZW46WE05UmJ3b0VNb05OUTN4MjdVTGNsSXYxbkNuXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### FFN 对比

- **Vs Transformer**

![Transformer 对比](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YjMwMmY0OTI2ZDQ0M2U5MDE2NzlkODc0OTQzNGFiMTNfY1Q3VlhRTGN3VjE4R05QVmF5UEoxbXpVR3hoeHRsb0dfVG9rZW46QU9ieWJvUjY1b3F4SU14c09Cb2NYbEd2bnJjXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

- **一般的 gating network 的计算，便于和 deepseek 做对比**

![Gating Network](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkyY2U0NDQ0MzUyYjlmYTg0M2RiMzI2ZTBiNzFiMWZfc1ZwOFpyS1NhdUUwNnJFbGZnUGZkZDRaZnYzb2VpelRfVG9rZW46VEdOQWJnTURab1V1alZ4b3BHQmNmSm5mbldiXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### 优势

- 相比 dense 模型，**预训练速度更快**
- 相比同参数量模型，**推理速度更快**
- 但是需要高 VRAM，因为所有专家都加载在内存中

## Switch Transformer

![Switch Transformer](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YmE1MzdjZjA0Y2M2YmJhNzZjNGI3NTEyNzE2NmRlZTBfd1FBOXByWnVhUmt1Tm1xanRTcm5ZU0ZKRnhRMTlKTVhfVG9rZW46Umw1dWJJdFdpb3JqcU14NlZsNWNKa2pSbmhkXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

# DeepSeek MOE(2024.01)

DeepSeek-V1 应该是 2023 年 12 月的 DeepSeek LLM Base 和 Chat 模型，是稠密模型。

DeepSeek-V2 及其之后的模型用的都是 MoE 了。

[DeepSeek MOE 原文](https://arxiv.org/pdf/2401.06066)

## 背景

- LLM 中，扩展模型参数时节约成本，故使用 MoE
- Deepseek MOE 就是为了通过更加高效的机制来确保专家之间的任务分配具有更高的专门化性
- 无法确保专家的专门化。这种重叠会导致专家没有获得足够的独特知识，也使得专家之间的差异化不明显，限制了模型的性能和效率
  - **知识混杂性（Knowledge Hybridity）**：在传统的 MoE 架构中，通常只使用有限数量的专家（例如 8 个或 16 个）。当某个 token 被分配给某个专家时，这些专家所涵盖的知识往往是多样化的，因此该专家的参数会试图同时存储和处理非常不同类型的知识。这种知识的多样性和复杂性导致专家的知识无法高度专注和聚焦，从而难以在同一模型中有效地利用这些不同类型的知识。
