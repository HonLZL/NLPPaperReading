# XLNet: Generalized Autoregressive Pretraining for Language Understanding

## 1 背景

Bert 依靠 mask 预训练，忽略了被 mask 位置的依赖关系，导致训练和预测是有差异的。

XLNet 将自回归融入进预训练



自回归  Auto Aggressive 和 自编码  Auto Encoder

<img src="note_images\image-20230222142635545.png" alt="image-20230222142635545" style="zoom: 67%;" />

GPT 是典型的自回归模型，缺点是忽略了上下文信息。Bert 是自编码模型，

XLNet 想融合 AR 和 AE 的优点。

### 2 Permutation Language Model 排列语言模型

XLNet 通过 Permutation 将 AR 和 AE 融合起来。

对一个长度为 T 的序列，那么这个序列将会有 T！个排列方式，从直觉上来说，如果模型的参数在所有的排列共享，那么模型将能够学习到双向的信息，$\mathbb{E}$ 表示 the expected likelihood。为验证这个想法: 
$$
\underset{\theta}{max} \ \ \  \mathbb{E}_{\textbf{z} \sim Z_T} \left [\sum_{t=1}^T log p_{\theta}(x_{z_t}|x_{\textbf{z}_{<t}})\right ]
$$
$Z_T$ 表示序列的所有排列组成的集合，$z \in Z_T$ , z 表示 其中一种排列。

$z_t$ 表示排列的第几个元素，$z_{<t}$ 表示排列 $z$ 的第 1 到第 $t-1$ 个元素。

$\theta$ 表示模型参数，$p_{\theta}$ 表示似然概率。

该公式，从所有排列中选取一个排列 $z$ ，计算条件概率并求和。参数 $\theta$ 在所有序列共享，因此 $x_t$ 看到了每个其他的元素。

由于排列太多，所以通常只随机采用部分序列。

论文中，并不是采用随机打乱序列的方式来实现 Permutation，而是使用原始序列的编码，依靠 transformer 的注意力掩码来实现分解顺序的置换，因为在微调时，只会遇到自然顺序的文本。

<img src="note_images\image-20230222170753934.png" alt="image-20230222170753934" style="zoom:67%;" />



例如原序列为 1234，随机抽取一种**预测顺序**为 3241，根据这个顺序进行 Attention Mask。

第一行：根据自回归的原理，1 可以看到 3 2 4，所以对应的圆圈不遮盖，圆圈标红。p(1|234)

第二行：2 只能看到 3，所以只有第 3 个圈标红。p(2|3)

第三行：3 啥都不能看到，因此没有标红。p(3)

第四行：4 可以看到 3 和 2，所以第三个和第二个标红。 p(4|32)

标红的意思是没被遮盖，白色代表被遮盖。

总体是 $p(x) = p(x_1|x_2x_3x_4)p(x_2|x_3)p(x_3)p(x_4|x_3x_2)$  





## 2 Two-Stream Self-Attention

### 2.1 回顾自回归与自编码

==自回归模型== 

给定文本序列 $\mathbf{x}=[x_1,x_2,···,x_T]$，语言模型的目标是调整参数 $\theta$ 使得训练数据上的似然函数最大：
$$
\max _{\theta}\ \log p_{\theta}(\mathbf{x})=\sum_{t=1}^{T} \log p_{\theta}\left(x_{t} \mid \mathbf{x}_{<t}\right)=\sum_{t=1}^{T} \log \frac{\exp \left(h_{\theta}\left(\mathbf{x}_{1: t-1}\right)^{T} e\left(x_{t}\right)\right)}{\sum_{x^{\prime}} \exp \left(h_{\theta}\left(\mathbf{x}_{1: t-1}\right)^{T} e\left(x^{\prime}\right)\right)}
$$
上式中 $\mathbf{x}_{<t}$ 表示 $t$ 时刻之前所有的 $x$，也就是 $\mathbf{x}_{1:t-1}$。$h_{\theta}( \mathbf{x}_{1:t-1})$ 表示 RNN 或者 transformer 在 t 之前的隐状态，$e(x)$ 是词 $x$ 的 embedding。



==自编码模型== 

BERT 通过将序列 $\mathbf{x}=[x_1,x_2,···,x_T]$ 中随机挑选 15% 的 Token 变成 [MASK] 得到带噪声版本的 $\hat{\mathbf{x}}$。假设被 Mask 的原始值为 $\overline{\mathbf{x}}$ ，那么 BERT 希望尽量根据上下文恢复（猜测）出原始值，也就是：
$$
\max _{\theta}\ \log p_{\theta}(\overline{\mathbf{x}} \mid \hat{\mathbf{x}}) \approx \sum_{t=1}^{T} m_{t} \log p_{\theta}\left(x_{t} \mid \hat{\mathbf{x}}\right)=\sum_{t=1}^{T} m_{t} \log \frac{\exp \left(H_{\theta}(\mathbf{x})_{t}^{T} e\left(x_{t}\right)\right)}{\sum_{x^{\prime}} \exp \left(H_{\theta}(\mathbf{x})_{t}^{T} e\left(x^{\prime}\right)\right)}
$$
上式中，若 $m_t=1$ 表示 $t$ 时刻是一个 Mask，需要恢复。$H_{\theta}$ 是一个 Transformer，它把长度为 $T$ 的序列 $\mathbf{x}$ 映射为隐状态的序列 $H_{\theta}(\mathbf{x})=[H_{\theta}(\mathbf{x})_1,H_{\theta}(\mathbf{x})_2,···，H_{\theta}(\mathbf{x})_T]$。注意：前面的语言模型的 RNN 在 $t$ 时刻只能看到之前的时刻，因此记号是 $h_{\theta}(\mathbf{x}_{1:t-1})$；而 BERT 的 Transformer（不同与用于语言模型的 Transformer）可以同时看到整个句子的所有 Token，因此记号是 $H_{\theta}(\mathbf{x})$ 。



==缺少位置信息== 

上述的排序语言模型的**自回归模型部分**存在位置信息缺失的问题。例如输入的句子是 $\text{I like New York}$, 并且采样到的一种排序为 $z = [1, 3, 4, 2]$, 在预测 $z_3 = 4$ 位置时，根据公式：

$$
p_\theta(X_{z_3} = x|x_{z_1z_2}) = p_\theta(X_4 = x|x_1x_3) = \frac{\exp(e(x)^Th_\theta(x_1x_3))}{\Sigma_{x'}\exp(e(x')^Th_\theta(x_1x_3))}
$$
上式用自然语言描述为，第一个词是 $\text{I}$, 第三个词是 $\text{New}$ 的条件下第四个词是 $\text{York}$ 的概率。

另外再看另一种排列 $z' = [1, 3, 2, 4]$, 在预测 $z_3 = 2$ 时：

$$
p_\theta(X_{z_3} = x|x_{z_1z_2}) = p_\theta(X_2 = x|x_1x_3) = \frac{\exp(e(x)^Th_\theta(x_1x_3))}{\Sigma_{x'}\exp(e(x')^Th_\theta(x_1x_3))}
$$
先不管预测的真实值是什么，先假设 $x$ 是 "York" 时的概率。则上式表示第一个词是 $\text{I}$，第三个词是 $\text{New}$ 的条件下第二个词是 $\text{York}$ 的概率。对比发现上面两个概率是相同的，这不符合日常经验。问题的关键就在于模型无法获知要预测的词在原始语句中的位置。

所以需要额外引入位置信息。预测位置是 $z_t$ 。$g_{\theta}$ 是一个新的模型，有两个输入信息，X 和 位置 z。
$$
p_{\theta}\left(X_{z_{t}}=x \mid \mathbf{x}_{z_{<t}}\right)=\frac{\exp \left(e(x)^{T} g_{\theta}\left(\mathbf{x}_{z_{<t}}, z_{t}\right)\right)}{\sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{T} g_{\theta}\left(\mathbf{x}_{z_{<t}}, z_{t}\right)\right)}
$$

### 2.2 双流信息分别是什么

通过 Attention 机制提取需要的信息，然后预测 $z_t$ 位置的词。那么它需要满足如下两点要求：

1. 为了预测 $\mathbf{x}_{z_t}$ ，$g_{\theta} (\mathbf{x}_{z_{<t}}, z_t)$ 只能使用位置信息 $z_t$ 而不能使用 $\mathbf{x}_{z_t}$。这是显然的：你预测一个词当然不能知道要预测的是什么词
2. 为了预测 $z_t$ 之后的词，$g_{\theta} (\mathbf{x}_{z_{<t}}, z_t)$ 必须编码了 $\mathbf{x}_{z_t}$ 的信息（语义）。

如下图，预测顺序是 [3,2,4,1]，对于 $g^{(1)}$ 层，要预测 $x_2$ 时，是不能编码 $x_2$ 的，要预测 $x_4$ 时，是不能编码 $x_4$ 的。但是因为下图左边和右边是同一层，$g_2^{(1)}$ 没有编码 $x_2$，预测 $x_4$ 又需要编码，这就出现了矛盾。

![image-20230308095200460](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230308095200460.png)

因此本文提出了 Two-Stream 去解决这个问题

1.**内容隐状态** Content stream

  $h_\theta (\mathbf{x}_{z_{<t}})$，简写为 $h_{z_t}$，和标准的 transformer 一样，既能编码上下文也能编码 $\mathbf{x}_{z_t}$ 的内容。token embedding

2.**查询隐状态** Queery stream

  $g_{\theta}(\mathbf{x}_{z_{<t}},z_t)$，简写为 $g_{z_t}$，它只编码上下文和要预测的位置 $z_t$，但是不包含 $\mathbf{x}_{z_t}$。trainable parameter

如下图，在预测 $x_2$ 时，Content stream 能编码 $x_2$ 和 $x_3$，Query stream 能编码 $x_3$，和 $x_2$ 的位置信息 $w$。

![image-20230308100207088](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230308100207088.png)

将上面两个结合起来是下图的 c

![image-20230307210220319](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230307210220319.png)

从第 1 层一直计算到 M 层。
$$
\begin{array}{l}
g_{z_{t}^{(m)}}^{(m)} \leftarrow \operatorname{Attention}\left(Q=g_{z_{t}}^{(m-1)}, K V=h^{(m-1)}_{{\color{Red} Z_{<t}}} ; \theta\right) (query \ stream:\ use\ z_t \ but \ cannot \ see\ x_{z_t} ) \\
h_{z_{t}}^{(m)} \leftarrow \operatorname{Attention}\left(Q=h_{z_{t}}^{(m-1)}, K V=h^{(m-1)}_{{\color{Red} Z_{\le t}}} ; \theta\right) (content\ stream:\ use\ both\ z_t\ and\ x_{z_t} ).
\end{array}
$$
上面的梯度更新和标准的 Self Attention 是一样的。在 fine-tuning 的时候，我们可以丢弃掉 Query 流而只用 Content 流

主要使用 LM，但是为了解决上下文的问题，引入了 Permutation LM。Permutation LM 在预测时缺乏 target 的位置信息，导致模型退化为 bag-od-words 模型，所以需要引入位置信息，但是会导致 预测自己和预测后面的词 的矛盾，因此通过引入 Two-Stream，Content 流编码到当前时刻的内容，而 Query  流只参考之前的历史以及当前要预测位置。最后为了解决计算量过大的问题，对于一个句子，我们只预测后 $\frac{1}{K}$ 个词。



## 3 Transformer-XL

### 3.1 [Transformer-XL](Transformer-XL.md) 的融入

如何将 Transformer-XL 的递归机制应用到 XL-net 的置换，并使模型能够重用以前片段中的隐藏状态。

设两个来自长度为 $T$ 的序列 $\mathbf{s}$ 的连续的片段为 $\tilde{\mathbf{x}}=\mathbf{s}_{1:T}$ 和 $\mathbf{x}=\mathbf{s}_{T+1:2T}$ 

设 $\tilde{\mathbf{z}}$ 和 $\mathbf{z}$ 分别是 [1,···,T] [T+1,···,2T] 的两个排列。

首先处理 $\tilde{\mathbf{z}}$ ， $\tilde{h}^{(m)}$ 是 $\tilde{\mathbf{z}}$ 第 $m$ 层的内容流的隐状态，需要缓存下来。知道了 $\tilde{\mathbf{z}}$ ，那么下一个片段 $\mathbf{z}$ 的内容流隐状态是：
$$
h_{z_t}^{(m)}\gets Attention(Q=h_{z_t}^{(m-1)},KV=\left [\mathbf{h}^{(m-1)}, \mathbf{h}_{\mathbf{z}_{\le t}}^{(m-1)} \right ]; \theta)
$$
KV 计算的 中括号表示维度拼接。

注意，位置编码只依赖于**原始序列**中的实际位置，因此，上述注意力更新一旦获得 $\tilde{h}^{(m)}$ ，就与 $\tilde{\mathbf{z}}$ 无关，这样就可以在不知道上一段分解顺序的情况下，对内存进行缓存和重用。

模型学习利用最后一个片段的所有排列的缓存信息，查询流用同样的方法运算。

### 3.2 Modeling Multiple Segments

  由于很多下游 NLP 任务中都包含了多个句子的情况，比如问答任务。下面我们讨论怎么在自回归框架下怎么预训练多个 segment。与训练阶段，和 BERT 一样，我们选择两个句子，它们是或者不是连续的句子（前后语义是否相关）。我们把这两个句子拼接后当成一个句子来学习 Permutation LM。和 BERT 一样，输入为 [CLS, A, SEP, B, SEP]，A、B 代表两个片段。根据消融实验，预训练不采用 NSP。

​	不同于 BERT，XLNet 采用相对片段编码。

BERT 使用的是绝对的 Segment 编码，也就是第一个句子对于的 Segment id 是 0，而第二个句子是 1。这样如果把两个句子换一下顺序，那么输出是不一样的。对于XLNet，给定序列一对位置 $i$, $j$ ，如果两个位置来自于同一个片段，就使用 segment encoding：$\mathbf{s}_{ij}=\mathbf{s}_{+}$ ,否则就是 $\mathbf{s}_{ij}=\mathbf{s}_{-}$ , $\mathbf{s}_{+}\ 和\ \mathbf{s}_{-}$是每个注意力头的可学习的参数。所以XLNet只考虑是否来自同一个片段，而不考虑到底来自于哪个片段，只对位置之间的关系建模。

当我们从位置 $i$ attend to $j$ 的时候，我们会这样计算一个新的 attention score：


$$
a_{ij}=(\mathbf{q}_i+\mathbf{b})^{\top} s_{ij}
$$
其中 $\mathbf{q}_i$ 是第 $i$ 个位置的 Query 向量，$\mathbf{b}$ 是一个可学习的 bias，$s_{ij}$ 是一个可学习的 embedding。最后我们会把这个 attention score 加到原来计算的 Attention score 里，这样它就能学到当 $i$ 和 $j$ 都属于某个 segment 的特征，以及 $i$ 和 $j$ 属于不同 segment 的特征。

## 4 总结

和语言模型相比，XLNet最大的优势就是通过输入序列的各种排列，同时学习到上下文的信息。







参考： https://wmathor.com/index.php/archives/1475/



























