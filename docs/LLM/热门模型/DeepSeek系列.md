---
article: false
title: DeepSeek系列
order: 2
---

## 问题和结论

1. MOE 能否做知识蒸馏？ <span style="color:blue;">**可以**</span>
2. MOE 相比 Dense 的优势？ <span style="color:blue;">**节约计算成本**</span>
3. 什么结构能用 MOE？  <span style="color:blue;">**任何 FFN → 多任务问题**</span>
4. 在已知 MOE 有负载不均衡问题的前提下，为啥目前大模型都开始抛弃传统 Transformer 架构，转投 MOE？ <span style="color:blue;">**便宜**</span>
5. 个人原来的理解：MOE 只能节约训练和推理的计算量，不能节约存储量；模型蒸馏可以节约计算量，也可以节约存储量，是否正确？ <span style="color:blue;">**正确**</span>

> **💡错误观点**：
>
> 1. MOE 是为了减小网络结构？ ❌ 相反，MOE的初衷是为了在保证较低计算量的同时，增加模型参数，使模型更强
> 2. MOE 是为了将深层网络变为浅层网络？ ❌ 将中间层参数数量从 N，降低为 N/E，分散到E个专家上，可能可以将网络变浅，但这不是主要目的
> 原来以为MOE是针对深层网络做的优化，将深层网络变为浅层网络，但是实际是


### DeepSeek 研究脉络

💡 **从稠密模型到混合专家，再到推理方向**

回顾 DeepSeek 过去一年多发表的核心论文，我们大致能将其研究分为两条主要脉络：　

- **基座模型（Foundation Models）**：从最早的 Dense（稠密）结构一路演进到 MOE（混合专家）模式，并在这个过程中不断发明和采用新的高效训练算法。
- **推理能力（Reasoning）**：包括解数学题、代码生成、逻辑问答乃至定理证明等，更强调大模型的"思考深度"，并在如何进行强化学习方面进行了连续多次创新。


## MOE 基本原理

MOE 全称是 Mixture of Experts，也就是混合专家模型

### 最最最原始版

#### 组成

1. **稀疏 MOE 层**：n 个专家 FFN
2. **路由**：token 到 top-K 个专家

#### FFN 对比

- **Vs Transformer**

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=OWZjMTVhMjhjZjZiZGQwNmU4MmMyYzRmNmE2OWU0NmZfeVBFYzdQUEFxSDB0Mmg5RG1hSFZsMGx0N045bGd5aHpfVG9rZW46QU9ieWJvUjY1b3F4SU14c09Cb2NYbEd2bnJjXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

![Transformer对比](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YjMwMmY0OTI2ZDQ0M2U5MDE2NzlkODc0OTQzNGFiMTNfY1Q3VlhRTGN3VjE4R05QVmF5UEoxbXpVR3hoeHRsb0dfVG9rZW46QU9ieWJvUjY1b3F4SU14c09Cb2NYbEd2bnJjXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

| ![FFn](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=N2JmZjZkZWJmNGIxMTU0ODRiYzc0MjkxMzVmMDYyNDRfc1JBcDQ5WW5yd2VoQ3ZWbXhlY2hJQUg3OXBXazJ5bk9fVG9rZW46UnA1M2Jja25Cb2c2MlJ4TTc5SGNtMFE1bmdlXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA) | ![MOE公式](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NzYxYTQzZGFjNzBiMTJkMTNhMTNjMzZiYTcwOGQ3YzZfZkx4TW1IQmlqWDNKWFQ0cFh2akNXNm16dFEwemxsZ2dfVG9rZW46WE05UmJ3b0VNb05OUTN4MjdVTGNsSXYxbkNuXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                             FFN                              |                             MOE                              |



- **一般的 gating network 的计算，便于和 deepseek 做对比**

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NjMyYjkxM2U2NDk3ZWE0NTYxYzVkYzZjOTI5NzQ5MzhfWVZqWXh0UmdxTDlZdk40bGEyYU56eER5SGdsUlZiNWlfVG9rZW46VEdOQWJnTURab1V1alZ4b3BHQmNmSm5mbldiXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

#### 优势

- 相比 dense 模型，**预训练速度更快**
- 相比同参数量模型，**推理速度更快**
- 但是需要高 VRAM，因为所有专家都加载在内存中

> 一个最直观的数据：
>
> 在 DeepSeek 官网上看到，DeepSeek-V3、V2.5 版本都用了 MoE 架构。但像 Qwen、LLama 模型，用的却是 Dense 架构，也就是传统的 Transformer 架构。这两种架构有个很明显的区别。DeepSeek-V3 版本总参数量高达 6710 亿，可每次计算激活的参数量，也就是真正参与到计算里的参数，只有 370 亿，是总参数量的 5.5%。但 Qwen 和 LLama 模型就不一样了，它们每次计算激活的参数量，就是整个模型的参数量，没有 “打折”。

### Switch Transformer

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=MDJhNzY4NTNmMzk0OTc0OTIwZDgwNTEwOTg0MjYyZDdfZjQ1YWN6UlBIaEk1NndFbGl3ajZncUNhUDV4amI1NWhfVG9rZW46Umw1dWJJdFdpb3JqcU14NlZsNWNKa2pSbmhkXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

## DeepSeek MOE(2024.01)

DeepSeek-V1 应该是 2023 年 12 月的 DeepSeek LLM Base 和 Chat 模型，是稠密模型。

DeepSeek-V2 及其之后的模型用的都是 MoE 了。

[DeepSeek MOE 原文](https://arxiv.org/pdf/2401.06066)

### 背景

- LLM 中，<span style="color:blue;">**扩展模型参数时节约成本**</span>，故使用 MoE

- Deepseek MOE 就是为了通过更加高效的机制来确保专家之间的任务分配具有更高的<span style="color:blue;">**专门化**</span>。
- 无法确保专家的专门化：这种重叠会导致专家没有获得足够的独特知识，也使得专家之间的差异化不明显，限制了模型的性能和效率。
  - **知识混杂性（Knowledge Hybridity）**：在传统的 MoE 架构中，通常只使用有限数量的专家（例如 8 个或 16 个）。当某个 token 被分配给某个专家时，这些专家所涵盖的知识往往是多样化的，因此该专家的参数会试图同时存储和处理非常不同类型的知识。这种知识的多样性和复杂性导致专家的知识无法高度专注和聚焦，从而难以在同一模型中有效地利用这些不同类型的知识。
  - **知识冗余性（Knowledge Redundancy）**：在 MoE 架构中，不同的专家可能需要共享相同的知识。当多个专家被分配到类似的任务时，它们可能会重复学习和存储相同的知识，这导致了多个专家之间的知识冗余，浪费了存储资源，同时也限制了专家在其各自领域的专门化，使其无法达到 MoE 模型的理论上限性能。

### 基本思想

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=MDE0OWJkMDRkYjE0NWFjOGE2OTZhOTVkMTg4YjQyNDVfejhISDdjZHV1WExFa3liOFpEOXk1T3RYMWdNMk9vem5fVG9rZW46WnlSY2JrWm1Tb1BwdTl4T01BR2NUVUJOblVoXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

#### **精细化专家划分**

> 划分更细，专家更加专业化，同时可以路由到更多的专家

在保持参数总量不变的情况下，我们通过拆分 FFN 的中间隐藏层维度来对专家进行更加精细的划分。同时，我们激活更多的精细化专家，从而实现更灵活、更适应的专家组合。精细化的专家划分允许多样化的知识更加细致地分解和学习，从而使每个专家能够专注于更高层次的专业化任务。专家激活的灵活性增加，也有助于更准确和针对性地获取知识。

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=MDMyZDYxYWRjNzU0ZTVlZTE3NGFjMWQ5MGZjZmM0N2ZfWkUzaVdXZjNCbmFKeDBxRUhzcHhFWllEZnJkemJLMVVfVG9rZW46VHZHUmJHTjZIbzhzRDF4MlNrcGMyT21HbnlnXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

#### **共享专家隔离**

我们将部分专家隔离出来，作为“共享专家”，始终被激活，用于捕捉和整合不同上下文中的共享知识。通过将共享知识压缩到这些共享专家中，减少了其他专家之间的冗余，从而提高了参数的效率，确保每个路由专家能够专注于独特的领域，保持高水平的专门化。

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWRlYmU4ZDNkZDM0OGM0NjNhMjg0MjY3ZmZmZDEwMGVfTjJLWmRSRHVtNTRUcjVHMlN6VTl5aFpiNzZaREtBMkJfVG9rZW46RWtXSGJhZW81b1FUcU14NFd3c2NuNEwwblBkXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=MTA0NDg0OTRlZmM4YjM5ZTM3YmI4ZDY2ZjAzNzBmZDVfd3JETnpNVDAxRUdQVUtPZUxyT2Jxc3F5MTJZYW1LRGRfVG9rZW46S0ZhYWJOMVp6b1d4bzd4cEs2T2M0Qks2bmRjXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

#### **负载均衡问题**

> MOE 类模型的通病
>
> 虽然稀疏门控能在不增加计算成本的情况下显著扩展模型参数空间，但其性能高度依赖门控机制的有效性。门控机制无法控制发给专家的 token 的概率，所以在实际操作中，会存在专家间工作负载分布不均衡的情况。某些专家被频繁使用（接收到了很多 token）而其他专家却很少被调用（接收的 token 寥寥无几）。这不仅不符合 MoE 的设计初衷（术业有专攻），还影响计算效率（例如引起分布式训练中各卡通讯时的负载不均）。

负载不均衡会造成：

1. 模型始终选择少数几个专家，其他专家缺乏充分训练，甚至部分专家参数完全没有更新
2. 专家并行计算时计算瓶颈（分到 16 张卡上，花了 16 张卡的运行时的钱，只有一张卡在工作）

解决方案：

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=M2YzZTY4NmMzMjNkN2VhMWQyYzQ1MTExNWIxZGQxZWJfbzUzN1cwVEt0OFU5Rm9QdXh6NDNKOXVpaDNCUVNyYzVfVG9rZW46VUZaYmJGZ2dxbzhNaHl4R3VtMWNLdUQ2bmhjXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

##### 专家级负载均衡

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=Njc4ODQyNTk3NjA2NDY5YTBjMGRmZWY2M2YwNmY1ZjZfYWZ4dFRlclA4a1RTVUpNNWpMUGtlN3BaeE5LeFdKSGhfVG9rZW46SXpaS2I5dUpDb0t0Umh4OE42RGNCYjNsbkdnXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjA5OTcxZTkwMWFmOTVjYzE0YmZjYzU5Y2I4NzQ5OThfbHlyZmU4a2xHZ2tvVmYzdEVnUXNrczFGb2xNdVQxZDdfVG9rZW46RnNEQ2I1MW9jb0U2Q1F4VWpZaWNIVW9ObjVmXzE3NDU1OTMyMjU6MTc0NTU5NjgyNV9WNA)

##### 设备级负载均衡





## DeepSeek-V2

进一步优化负载均衡





## DeepSeek-V3(Reasoning model)

1. 门控函数优化

> 首先 V3 的模型远大于 V2，V3 的每层 MOE 中有 256 个路由专家，8 个激活专家。但 V2 中只有 160 个路由专家，6 个激活专家，从参数上就可以发现 V3 的门控函数计算量远大于 V2，大家也都清楚当计算维度变大时 SoftMax 的前向和反向是很耗费计算资源的，而 Sigmod 直接将数值映射到[0，1]之间，相对来说更加简单。可能实现效果也类似，因此为了更加高效的训练从而进行了替换。

1. 进一步优化负载均衡
   1. 无辅助损失的负载均衡
   2. 互补序列层面的辅助损失



## 参考资料

- [MOE 介绍](https://kevincheung2259.github.io/2024/09/13/MOE-Intro/index.html)
- [DeepSeek 技术解析](https://deepseek.csdn.net/67fa2941da5d787fd5cb6acb.html)
- [负载均衡部分参考资料](https://www.cnblogs.com/rossiXYZ/p/18835426#0x00-概述)
