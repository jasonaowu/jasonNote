---
article: false
title: DeepSeek系列
order: 2
---

# 问题和结论

1. MOE 能否做知识蒸馏？  可以
2. MOE 相比 Dense 的优势？  节约计算成本
3. 什么结构能用 MOE？  任何 FFN   --> 多任务问题
4. 在已知 MOE 有负载不均衡问题的前提下，为啥目前大模型都开始抛弃传统 Transformer 架构，转投 MOE？   便宜
5. 个人原来的理解：MOE 只能节约训练和推理的计算量，不能节约存储量；模型蒸馏可以节约计算量，也可以节约存储量，是否正确？  正确

> 错误观点：
>
> 1. MOE 是为了减小网络结构？ ❌，相反，MOE的初衷是为了在保证较低计算量的同时，增加模型参数，使模型更强
>
> 原来以为MOE是针对深层网络做的优化，将深层网络变为浅层网络，但是实际是将中间层参数数量从 N ，降低为 N/E，分散到E个专家上

与稠密模型相比，对于给定的计算预算，MoE 模型提供更高效的训练。这是因为门控网络仅将 token 发送到一部分专家，从而减少了计算负载。因此，模型的容量（其参数总数）可以增加，而不会成比例地增加计算需求。在推理期间，仅使用部分专家，因此 MoE 能够执行比稠密模型更快的推理。但是，整个模型需要加载到内存中，而不仅仅是正在使用的专家。

MoE 中实现更高计算效率的稀疏性来自于这样一个事实：特定的 token 只会被路由到一部分专家。专家的数量以及如何选择专家取决于门控网络的实现，但一种常见的方法是 top k。门控网络首先预测每个专家的概率值，然后将 token 路由到 top k 个专家以获得输出。但是，如果所有 token 始终都发送到相同的专家子集，则训练效率会降低，而其他专家最终会训练不足。为了缓解这个问题，引入了负载均衡损失，以鼓励均匀路由到所有专家。

专家的数量和选择 top k 个专家是设计 MoE 的重要因素。更多的专家数量允许扩展到更大的模型，而不会增加计算成本。这意味着模型具有更高的学习能力，但是，超过某个点后，性能增益往往会减少。选择的专家数量需要与服务模型的推理成本相平衡，因为整个模型都需要加载到内存中。同样，在选择 top k 时，训练期间较低的 top k 会导致较小的矩阵乘法，如果通信成本足够大，则会浪费计算资源。但是，在推理期间，较高的 top k 通常会导致较慢的推理速度。

https://kevincheung2259.github.io/2024/09/13/MOE-Intro/index.html

https://deepseek.csdn.net/67fa2941da5d787fd5cb6acb.html

以Transformer原文中FFN部分的参数量进行计算，如果换成MOE架构，参数量是多少，为啥能够节约计算时间？

💡从稠密模型到混合专家，再到推理方向

回顾 DeepSeek 过去一年多发表的核心论文，我们大致能将其研究分为两条主要脉络：　

- **基座模型（Foundation Models）**：从最早的 Dense（稠密）结构一路演进到 MOE（混合专家）模式，并在这个过程中不断发明和采用新的高效训练算法。
- **推理能力（Reasoning）**：包括解数学题、代码生成、逻辑问答乃至定理证明等，更强调大模型的“思考深度”，并在如何进行强化学习方面进行了连续多次创新。

在阅读这份逐篇解读之前，可以先记住 DeepSeek 的几大特色：对实验和数据极度重视、有足够的冒险精神尝试新架构和新算法、且真正愿意分享内部研究细节，为社区提供可复现的技术报告。　

# MOE 基本原理

MOE全称是Mixture of Experts，也就是混合专家模型

## 最最最原始版

组成：

1. 稀疏MOE层：n个专家FFN
2. 路由：token到top-K个专家。

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NWVjNjk1ZmU5YmUyNDA3ZGZlZGQ3MzljZWU5NzVhZWJfQ1pESDhYRFRIM1QwNVEwTzRJSVNCdWZOS1FkNHJ5Q2JfVG9rZW46UnA1M2Jja25Cb2c2MlJ4TTc5SGNtMFE1bmdlXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NzMzNzMyZWE5ZDFkYjc1MTUwODE0OTViZjI4MWFjNGNfdTRrMVc0UXJnemtRekFWNTNhYzZwMXRBc2plak9naWZfVG9rZW46WE05UmJ3b0VNb05OUTN4MjdVTGNsSXYxbkNuXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

FFN

- Vs Transformer

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YjMwMmY0OTI2ZDQ0M2U5MDE2NzlkODc0OTQzNGFiMTNfY1Q3VlhRTGN3VjE4R05QVmF5UEoxbXpVR3hoeHRsb0dfVG9rZW46QU9ieWJvUjY1b3F4SU14c09Cb2NYbEd2bnJjXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

- 一般的gating network的计算，便于和deepseek做对比

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkyY2U0NDQ0MzUyYjlmYTg0M2RiMzI2ZTBiNzFiMWZfc1ZwOFpyS1NhdUUwNnJFbGZnUGZkZDRaZnYzb2VpelRfVG9rZW46VEdOQWJnTURab1V1alZ4b3BHQmNmSm5mbldiXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### 优势

- 相比dense模型，预训练速度更快
- 相比同参数量模型，推理速度更快
- 但是需要高 VRAM，因为所有专家都加载在内存中

## Switch Transformer

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YmE1MzdjZjA0Y2M2YmJhNzZjNGI3NTEyNzE2NmRlZTBfd1FBOXByWnVhUmt1Tm1xanRTcm5ZU0ZKRnhRMTlKTVhfVG9rZW46Umw1dWJJdFdpb3JqcU14NlZsNWNKa2pSbmhkXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

# DeepSeek MOE(2024.01)

DeepSeek-V1 应该是2023年12月的 DeepSeek LLM Base 和 Chat 模型，是稠密模型

DeepSeek-V2 及其之后的模型用的都是MoE了

[DeepSeek MOE原文](https://arxiv.org/pdf/2401.06066)

## 背景

- LLM中，扩展模型参数时节约成本，故使用MoE
- Deepseek MOE就是为了通过更加高效的机制来确保专家之间的任务分配具有更高的专门化性。
- 无法确保专家的专门化。这种重叠会导致专家没有获得足够的独特知识，也使得专家之间的差异化不明显，限制了模型的性能和效率。
  - **知识混杂性（Knowledge Hybridity）**：在传统的MoE架构中，通常只使用有限数量的专家（例如8个或16个）。当某个token被分配给某个专家时，这些专家所涵盖的知识往往是多样化的，因此该专家的参数会试图同时存储和处理非常不同类型的知识。这种知识的多样性和复杂性导致专家的知识无法高度专注和聚焦，从而难以在同一模型中有效地利用这些不同类型的知识。
  - **知识冗余性（Knowledge Redundancy）**：在MoE架构中，不同的专家可能需要共享相同的知识。当多个专家被分配到类似的任务时，它们可能会重复学习和存储相同的知识，这导致了多个专家之间的知识冗余，浪费了存储资源，同时也限制了专家在其各自领域的专门化，使其无法达到 MoE 模型的理论上限性能。

## 基本思想

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NTE3MTcxZjZkMDM1YzhlZDFiMmExOWUwODhjNTY5YzZfN2E5Z1JOZEZ0UDN6WmR1VFJjc0tmVjhKcnBoTExaS1ZfVG9rZW46WnlSY2JrWm1Tb1BwdTl4T01BR2NUVUJOblVoXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### **精细化专家划分**

> 划分更细，专家更加专业化，同时可以路由到更多的专家

在保持参数总量不变的情况下，我们通过拆分FFN的中间隐藏层维度来对专家进行更加精细的划分。同时，我们激活更多的精细化专家，从而实现更灵活、更适应的专家组合。精细化的专家划分允许多样化的知识更加细致地分解和学习，从而使每个专家能够专注于更高层次的专业化任务。专家激活的灵活性增加，也有助于更准确和针对性地获取知识。

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=NTRlZGY1ODQ0MWU2OGNkMWVhMzZmZjUzZDExNmVjZGNfdTdGTmxHOFhOdjhRVGhVTzJ6ZFdYd01GZGNCZXg1TTVfVG9rZW46VHZHUmJHTjZIbzhzRDF4MlNrcGMyT21HbnlnXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### 共享专家隔离

我们将部分专家隔离出来，作为“共享专家”，始终被激活，用于捕捉和整合不同上下文中的共享知识。通过将共享知识压缩到这些共享专家中，减少了其他专家之间的冗余，从而提高了参数的效率，确保每个路由专家能够专注于独特的领域，保持高水平的专门化。

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=MjA5N2NhNjE1Y2IzNWE2YmI2NTZhZjcyZjM4MTg3ZmVfeVdSZFRaVmY0UVVnNW1nNlpLejNSU090UGJYVFlYZ0xfVG9rZW46RWtXSGJhZW81b1FUcU14NFd3c2NuNEwwblBkXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YjA5MzYyNjdmYWQ3NzlhNWIzNGI5NTljZTc1Yzg0YzNfTUI2Wk1mN2hORER5MktlUEJHdEFoVGpOWGNZOFJyaUJfVG9rZW46S0ZhYWJOMVp6b1d4bzd4cEs2T2M0Qks2bmRjXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

### 负载均衡问题

> MOE类模型的通病
>
> 虽然稀疏门控能在不增加计算成本的情况下显著扩展模型参数空间，但其性能高度依赖门控机制的有效性。门控机制无法控制发给专家的token的概率，所以在实际操作中，会存在专家间工作负载分布不均衡的情况。某些专家被频繁使用（接收到了很多token）而其他专家却很少被调用（接收的token寥寥无几）。这不仅不符合MoE的设计初衷（术业有专攻），还影响计算效率（例如引起分布式训练中各卡通讯时的负载不均）。

负载不均衡会造成：

1. 模型始终选择少数几个专家，其他专家缺乏充分训练，甚至部分专家参数完全没有更新
2. 专家并行计算时计算瓶颈（分到16张卡上，花了16张卡的运行时的钱，只有一张卡在工作）

解决方案：

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YjhmMTA5MzAyZDM1NmNlNDhiYmU3YTA0NzEwZTI2MTRfMTBpQjZqU3FIV0lTR1Q1MDRtMnNuN3h5NzFnWEZrU2tfVG9rZW46VUZaYmJGZ2dxbzhNaHl4R3VtMWNLdUQ2bmhjXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

#### 专家级负载均衡

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=YmNjZjA4YzM1MTQxNjBkYmIwMjZmODNjYmQ3NjEyODNfczVBZ2FQUzd6ZUY2ZUlqMTBxcmNQZWIyaDZzdFNlWGpfVG9rZW46SXpaS2I5dUpDb0t0Umh4OE42RGNCYjNsbkdnXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

![img](https://s08a4grxpw8.feishu.cn/space/api/box/stream/download/asynccode/?code=Mzk3NjM5NDg1YzE5NDVjMGIyODk4ZGZmYjMwM2U0ZWVfY3VBSnllODZ4VnpHcDJCZldzQ1lIOHJRdmp0cUhjRHJfVG9rZW46RnNEQ2I1MW9jb0U2Q1F4VWpZaWNIVW9ObjVmXzE3NDU1ODU4NTk6MTc0NTU4OTQ1OV9WNA)

#### 设备级负载均衡

## 参考

[负载均衡部分参考资料](https://www.cnblogs.com/rossiXYZ/p/18835426#0x00-概述)

# DeepSeek-V2

进一步优化负载均衡

# DeepSeek-V3(Reasoning model)

1. 门控函数优化

> 首先V3的模型远大于V2，V3的每层MOE中有256个路由专家，8个激活专家。但V2中只有160个路由专家，6个激活专家，从参数上就可以发现V3的门控函数计算量远大于V2，大家也都清楚当计算维度变大时SoftMax的前向和反向是很耗费计算资源的，而Sigmod直接将数值映射到[0,1]之间，相对来说更加简单。可能实现效果也类似，因此为了更加高效的训练从而进行了替换。

1. 进一步优化负载均衡
   1. 无辅助损失的负载均衡
   2. 互补序列层面的辅助损失

671B的DeepSeek R1，750G的硬盘都671B的权重都塞不下

上图中(a)表示之前的MOE架构，专家分的粒度比较粗，并且没有共享专家，图（b）是将专家粒度划分的更细情况，图（c）在图(b)的基础上增加了共享专家。

在 DeepSeek 官网上看到，DeepSeek-V3、V2.5 版本都用了 MoE 架构。但像 Qwen、LLama 模型，用的却是 Dense 架构，也就是传统的 Transformer 架构。这两种架构有个很明显的区别。DeepSeek-V3 版本总参数量高达 6710 亿，可每次计算激活的参数量，也就是真正参与到计算里的参数，只有 370 亿，是总参数量的 5.5%。但 Qwen 和 LLama 模型就不一样了，它们每次计算激活的参数量，就是整个模型的参数量，没有 “打折”。为啥会出现这种差异呢？

MoE 模型与传统大模型的典型区别：

MoE 模型：每次输入时，只会激活一小部分专家（例如，10% 的专家），而其他专家不参与计算。这意味着，MoE 模型可以在保持模型参数量很大的情况下，大幅度减少计算量，提高了计算效率和资源利用。

传统大模型：在传统的大型神经网络（如 Transformer）中，所有层和所有节点在每次前向传播时都会参与计算。虽然这些模型参数也可能非常庞大，但每次输入都需要对所有的参数进行计算，即使部分参数的贡献很小，因此也会浪费计算资源。

MoE 主要的变化点在 前馈网络（FFN） 层，它被 MoE 机制取代，包括：

1. 专家网络（Experts）：多个前馈网络（FFN），相当于多个可选的专家，每个专家结构类似于普通 FFN。
2. 门控网络（Gating Network）：决定在每次输入时，选择哪些专家进行计算，并分配权重。
3. 专家混合（Mixture of Experts）：选定的专家执行计算，并对其输出进行加权合并。

其具体工作流程如下：

1. 输入 token（与传统大模型一致）：

输入的 token 会首先经过标准的 token embedding 和 位置编码 处理，转化为对应的向量表示。

1. 多头自注意力（MHSA）层 （与传统大模型一致）

在多头自注意力层中，token 会计算自己与其他 token 的注意力权重，捕获序列中的长距离依赖关系。这一层的输出是增强了上下文信息的 token 表示，每个 token 在经过多头自注意力层后，会有一个上下文信息丰富的表示，通常是一个向量 h（比如维度为 768 或 1024）。这个向量包含了该 token 在整个句子中上下文的信息，反映了这个 token 和其他 token 的关系。

1. MoE 层（变化的地方，包含门控网络、专家网络、专家混合）

[混合专家模型 (MoE) 详解](https://v11enp9ok1h.feishu.cn/wiki/YC1bwhcyJiQhuJksgO7cVKmqn8e#:~:text=本文讨论了混合专家模型（MoE）的相关内容，包括其与稠密模型相比的优势、结构组成、训练和推理挑战、发展简史、解决问题的方法、不同模型的特点、微调策略、适用场景、优化方法、开源项目以及研究方向等。 关键要点包括： 1.,模型优势：与稠密模型相比，预训练速度更快，相同参数数量下推理速度更快，但需大量显存。 2. 结构组成：由稀疏MoE层和门控网络或路由组成，MoE层含若干专家，门控网络决定令牌路由。)

https://zhuanlan.zhihu.com/p/21584562624

https://zhuanlan.zhihu.com/p/21584562624 参考

MOE的基本原理是使用混合专家来替代原transformer架构中的前向反馈层（FFN），在论文中的示意图如下：

https://arxiv.org/pdf/2101.03961

截取下原始论文的参数变化图和框架图

https://zhuanlan.zhihu.com/p/18565423596

DeepSeek 在 2T token 上训练了 DeepSeekMoE 16B，激活参数量 2.8B，仅使用了 DeepSeek 7B 和 LLaMA 2 7B 约 40% 的计算量，但评测性能相当。

https://www.armcvai.cn/2025-02-12/deepseek-moe-code.html

https://developer.volcengine.com/articles/7476296702404591654   token 的详细解释

包含 236B 参数，其中每个 token 激活 21B 参数，并支持 128K tokens 的上下文长度。DeepSeek-V2 采用了创新的架构，包括多头潜在注意力（MLA）和 DeepSeekMoE

 

与 DeepSeek 67B 相比，DeepSeek-V2 实现了显著更强的性能，同时节省了 42.5% 的训练成本，减少了 93.3% 的 KV 缓存，并将最大生成吞吐量提升了 5.76 倍。

https://www.youtube.com/watch?v=0BodppoiloM&ab_channel=chaofa%E7%94%A8%E4%BB%A3%E7%A0%81%E6%89%93%E7%82%B9%E9%85%B1%E6%B2%B9

https://www.youtube.com/watch?v=P7txFafuUOE&t=161s&ab_channel=EZ.EncoderAcademy

https://arxiv.org/pdf/2401.06066

 

https://www.youtube.com/watch?v=G1vC1gjcJEI&ab_channel=TensorOps

https://www.youtube.com/watch?v=pl38wKk-dHo&ab_channel=AILinkDeepTech

https://www.youtube.com/watch?v=bd2U-OJ7UJc&ab_channel=GaspardBaye

https://zhuanlan.zhihu.com/p/18565423596

https://cloud.tencent.cn/developer/article/2505656?policyId=1004

为了比较原始 14 层堆叠 Autoencoder 与 2 层 MoE + Autoencoder 的参数差异，我们基于以下假设进行结构化分析：

1. **原始 14 层堆叠 Autoencoder 的参数计算**

- **结构假设**：对称编码器-解码器结构，每部分 7 层，逐步降维至 4 维（编码器）后逐步恢复（解码器）。
- **参数计算**（以全连接层为例）：
  - **编码器路径**： `3000 → 1500 → 750 → 375 → 188 → 94 → 47 → 4` 每层参数：`输入维度×输出维度 + 输出维度`（含偏置）。 编码器总参数：**≈6.0M**
  - **解码器路径**： `4 → 47 → 94 → 188 → 375 → 750 → 1500 → 3000` 解码器总参数：**≈6.0M**
  - **总参数**：编码器 + 解码器 ≈ **12.0M**

1. **2 层 MoE + Autoencoder 的参数计算**

- **结构假设**：
  - **编码器**：1 层 MoE（含 2 个专家），直接降维至 4 维。
  - **解码器**：1 层 MoE（含 2 个专家），从 4 维恢复至 3000 维。
  - **专家结构**：每个专家为单层全连接网络。
- **参数计算**：
  - **编码器 MoE 层**：
    - 每个专家参数：`3000×4 + 4 = 12,004`
    - 2 个专家总参数：`2×12,004 = 24,008`
    - 门控网络参数：`3000×2 + 2 = 6,002`
    - 编码器总参数：**30,010**
  - **解码器 MoE 层**：
    - 每个专家参数：`4×3000 + 3000 = 15,000`
    - 2 个专家总参数：`2×15,000 = 30,000`
    - 门控网络参数：`4×2 + 2 = 10`
    - 解码器总参数：**30,010**
  - **总参数**：编码器 + 解码器 ≈ **60,020**

1. **参数减少对比**

| 模型                 | 总参数      | 参数减少比例 |
| -------------------- | ----------- | ------------ |
| 14层堆叠Autoencoder  | ~12,000,000 | -            |
| 2层MoE + Autoencoder | ~60,020     | 99.50%       |

1. **关键结论**

- **参数锐减原因**： MoE 通过稀疏激活（仅需少量专家参与计算），结合极简结构设计（单层编码/解码），大幅压缩参数量。
- **潜在代价**： 模型容量可能下降，需通过知识蒸馏或专家优化弥补性能损失。
- **适用场景**： 资源受限环境（如边缘设备），需轻量化模型且能容忍一定精度损失。

1. **扩展讨论**

若需平衡性能与参数效率，可尝试：

1. **增加专家数量**（如 4 个专家），参数仍远低于原始模型（约 120k vs 12M）。
2. **混合结构**：部分层用 MoE，其余保留全连接层，灵活调节参数与性能。
3. **知识蒸馏**：用原始 14 层模型作为教师，指导 2 层 MoE 学生模型进一步优化性能。

以下是原始 14 层堆叠 Autoencoder 与 2 层 MoE+Autoencoder 在推理时间、训练时间和内存占用的详细对比分析：

1. **推理时间对比**

| 模型                | 计算复杂度 | 实际推理速度 | 关键影响因素                                                 |
| ------------------- | ---------- | ------------ | ------------------------------------------------------------ |
| 14层堆叠Autoencoder | 高         | 较慢         | - 深层全连接结构，需逐层计算。 - 参数总量大（12M），计算密集。 |
| 2层MoE+Autoencoder  | 低         | 较快         | - 仅2层MoE结构，计算步骤少。 - 稀疏激活（每次仅调用少量专家），实际计算量远低于参数总量。 - 门控网络引入额外计算，但总体仍显著节省时间。 |

**结论**：

- **MoE+Autoencoder 推理更快**，得益于稀疏激活和极简层数，尤其适合实时推理场景（如边缘设备）。

1. **训练时间对比**

| 模型                | 收敛速度 | 单批次训练时间 | 总训练成本 | 关键影响因素                                                 |
| ------------------- | -------- | -------------- | ---------- | ------------------------------------------------------------ |
| 14层堆叠Autoencoder | 较慢     | 长             | 高         | - 深层网络梯度传递复杂，易出现梯度消失/爆炸。 - 参数多（12M），反向传播计算量大。 |
| 2层MoE+Autoencoder  | 较快     | 短             | 低         | - 层数少，梯度传递直接。 - 参数少（60k），反向传播高效。 - 需额外优化门控网络与专家负载平衡，可能略微增加调参成本。 |

**结论**：

- **MoE+Autoencoder 训练总时间更短**，但需注意门控网络的稳定性（如专家利用率均衡）。

1. **内存占用对比**

#### **3.1 训练阶段内存占用**

| 模型                | 参数内存 | 梯度内存 | 优化器状态内存  | 总内存（float32） |
| ------------------- | -------- | -------- | --------------- | ----------------- |
| 14层堆叠Autoencoder | 46 MB    | 46 MB    | 138 MB（Adam）  | ~230 MB           |
| 2层MoE+Autoencoder  | 0.23 MB  | 0.23 MB  | 0.69 MB（Adam） | ~1.15 MB          |

#### **3.2 推理阶段内存占用**

| 模型                | 参数内存（float32） | 激活值内存     | 总内存      |
| ------------------- | ------------------- | -------------- | ----------- |
| 14层堆叠Autoencoder | 46 MB               | 高（14层激活） | ~100-200 MB |
| 2层MoE+Autoencoder  | 0.23 MB             | 低（2层激活）  | ~10-20 MB   |

**关键说明**：

- **MoE+Autoencoder 内存需求极低**，尤其适合内存受限场景（如移动端部署）。
- 激活值内存差异显著：深层模型需存储多层中间结果，而 MoE 模型仅需少量层激活。

1. **综合对比表**

| 指标     | 14层堆叠Autoencoder | 2层MoE+Autoencoder | 优势方          |
| -------- | ------------------- | ------------------ | --------------- |
| 参数量   | ~12.0M              | ~60k               | MoE+Autoencoder |
| 推理时间 | 慢                  | 快                 | MoE+Autoencoder |
| 训练时间 | 长                  | 短                 | MoE+Autoencoder |
| 训练内存 | ~230 MB             | ~1.15 MB           | MoE+Autoencoder |
| 推理内存 | ~100-200 MB         | ~10-20 MB          | MoE+Autoencoder |
| 模型容量 | 高                  | 较低               | 堆叠Autoencoder |
| 适用场景 | 高精度需求          | 资源受限环境       | 场景依赖        |

1. **权衡与建议**

- **选择 14 层堆叠 Autoencoder**： 需高精度且资源充足（如云端训练/推理），容忍较高延迟和内存消耗。
- **选择 2 层 MoE+Autoencoder**： 资源受限（如嵌入式设备）、需快速响应或低内存占用，可接受轻微精度损失。
- **优化方向**： 若需兼顾性能与效率，可尝试：
  - **知识蒸馏**：用 14 层模型指导 MoE 模型提升精度。
  - **混合结构**：部分层使用 MoE，平衡参数与容量。
  - **动态专家数量**：根据输入复杂度调整激活专家数。

1. **总结**

- **MoE+Autoencoder 在效率上全面占优**，参数、内存、时间均降低 1～2 个数量级。
- **堆叠 Autoencoder 在模型容量上占优**，适合对精度要求严格的场景。
- 实际选择需结合任务需求、硬件条件及精度-效率权衡。
