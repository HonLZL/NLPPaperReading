# GeoLayoutLM: Geometric Pre-training for Visual Information Extraction

## 1 背景

现有模型大多以隐式的方式学习布局信息，这对 RE 任务是不够的。

GeoLayoutLM显式地对预训练中的几何关系进行建模，我们称之为几何预训练。 



LayoutLMv3 的缺点

与几何布局相比，布局 LayoutLMv3 倾向于将两个实体更多地依赖其语义进行链接

几何关系是描述文档布局的一种具体形式，对于文档布局表示具有重要意义 。



即使召回率不变，只使用简单的限制实体之间的距离，precision 也会提高 4 个点。

|                        | Precision | Recall | F1    |
| ---------------------- | --------- | ------ | ----- |
| LayoutLMv3             | 75.82     | 85.45  | 80.35 |
| + geometric constraint | 79.87     | 85.45  | 82.57 |



![image-20230719144936278](note_images\image-20230719144936278.png)

左图红箭头为 LayoutLMv3 预测错误的关系。

右图 LayoutLMv3 预测对了上面的，但是下面的却漏掉了。

因此，有必要通过在预训练过程中显式地建模实体间的几何关系，为文档预训练模型学习更好的布局表示。 



​	在RE微调过程中，以往工作通常从头学习单个线性或双线性层这样的任务头。一方面，由于文档中超出 token 或 text-segment 特征的 higher-level pair 关系特征是复杂的，我们认为单一的线性或双线性层并不总是足以充分利用RE的编码特征。另一方面，随机初始化的RE任务头在有限的微调数据下容易出现过拟合现象。 

​	既然预训练的骨干已经表现出了巨大的潜力，为什么不同时以某种方式预训练任务头呢? 一些工作已经证明，预训练和微调之间的差距越小，下游任务的性能越好。 



本文采用注重几何的训练策略和精心设计的RE预训练任务头。

将几何关系从一对拓展为多个。





## 2 模型

**GeoLayoutLM** 是一个面向VIE的多模态框架。



### 2.1 模型结构

![image-20230719162540112](note_images\image-20230719162540112.png)



分为左右两个模块，**Vision Module** 和 **Text-Layout Module** 

按照 LayoutLMv3，text embedding 使用 5 个 embeddings 的相加，分别为 

**token embeddings** 

**1D position embeddings** 

**1D segment rank embeddings **

**1D segment BIE embeddings **

**2D segment box embeddings** 

**Vision module** 的输出的特征通过 **全局池化** 和 **RoI 对齐**，获得全局视觉特征 $F_{v0}$，和 $n$ 个视觉片段特征 $\{ F_{vi}|i\in[1,n] \}$。然后视觉协同注意力模块(visual co-attention module)将 $\{F_{vi}\}$ 作为 query，来自 Text-Layout Module 的 $\{F_{ti}\}$ 作为 key 和 value，然后输出混合的视觉特征 $\{M_{vi}\}$。反过来，将  $\{F_{ti}\}$ 作为 query，将 $\{F_{vi}\}$  作为 key 和 value，可以计算得到混合文本特征 $\{M_{ti}\}$。

最终，将 $\{M_{vi}\}$ 和 对应的片段的第一个 token 特征 $\{M_{t,b(i)}\}$ 相加得到 第 i 个片段的特征 $H_i$。



### 2.2 Relation Heads

在微调时，对于 SER 任务，用一个简单的 MLP 分类层是有效的，但是对于 RE 任务，简单的线性层或者双线性层是不足够的。

本文提出两个 relation heads 去增强 关系预训练 和 RE 微调 的关系特征表示，分别是：

Coarse Relation Prediction（CPR）head 和 Relation Feature Enhancement (RFE) head

REF 头是一个轻量的 transformer，其中有一个标准的 encoder 层，丢弃自注意力机制的的 decoder （为了计算效率），全连接层后有一个 sigmoid 激活函数层。

![image-20230720173047071](note_images\image-20230720173047071.png)

