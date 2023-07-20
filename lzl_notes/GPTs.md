# GPTs

## 1 GPT1

Improving Language Understanding by Generative Pre-Training

### 1 背景

NLP 对依赖于监督学习，这需要大量人工标注数据，限制了 NLP 模型在缺乏标注的领域的适用性。在这种情况下，模型要是能够利用未标注的数据就好了。此外，即使在监督学习的情况下，无监督也能够提供显著的性能提升，例如分词，此 embedding等。

**难点** 

1. 没有一个优化目标，无法计算损失
2. 用未标注的数据训练后，不知道如何将学习到的迁移到目标任务

 

### 2 模型

#### 2.1 自监督预训练 

给定一个 无监督的 tokens 语料库 $\mathcal{U}=\{u_1,u_2,...,u_n  \}$，
$$
L_{1}(\mathcal{U})=\sum_{i} \log P\left(u_{i} \mid u_{i-k}, \ldots, u_{i-1} ; \Theta\right)
$$
$k$ 是 文本窗口，$P$ 是参数为 $\Theta$  的概率分布，这两个都是可训练参数。

然后使用 多层 transformer 的 decoder：
$$
\begin{aligned}
h_{0} & =U W_{e}+W_{p} \\
h_{l} & =\operatorname{transformer} b \operatorname{block}\left(h_{l-1}\right) \forall i \in[1, n] \\
P(u) & =\operatorname{softmax}\left(h_{n} W_{e}^{T}\right)
\end{aligned}
$$
$U=(u_{-k}, ...,u_{-1})$ 是 tokens 的向量，$n$ 是 decoder 的层数，$W_e$ 是 token embedding 矩阵, $W_p$ 是position embedding矩阵



#### 2.2 有监督微调 

数据集为 $\mathcal{C}$，每个实例包含一个 token 序列 $x^1,x^2,...,x^m$，label 为 $y$，transformer decoder最后的激活层为 $h_l^m$，然后喂进线性输出层 $W_y$ 去预测 $y$： 
$$
P\left(y \mid x^{1}, \ldots, x^{m}\right)=\operatorname{softmax}\left(h_{l}^{m} W_{y}\right)
$$
最大化目标为：
$$
L_{2}(\mathcal{C})=\sum_{(x, y)} \log P\left(y \mid x^{1}, \ldots, x^{m}\right)
$$
实验发现，加入语言模型学习目标作为辅助任务，能够：1.提升监督模型的泛化能力；2.加快收敛。总的最大化目标为：
$$
L_{3}(\mathcal{C})=L_{2}(\mathcal{C})+\lambda*L_{1}(\mathcal{C})
$$
因此只需要微调 $W_y$ 和特殊 Token

### 3 微调任务

![image-20230511205556539](note_images\image-20230511205556539.png)



实验

7000本未发表的书





## 2 GPT2

Language Models are Unsupervised Multitask Learners



### 2.1 背景



















