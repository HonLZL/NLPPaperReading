# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## 1 背景

Transformers 虽然具有学习长期依赖关系的潜力，但在语言建模的设置上受到固定长度上下文的限制。比如原 transformer限制为 512 ，

如何学习长期一代一直是一个长期存在的研究问题，以往都是修改 RNN 的内部架构去化解梯度消失。然而本文基于 transformer。 



**Vanilla model** 

应用 Transformer 或者 self-attention，存在一个问题是如何训练Transformer将任意长的上下文有效地编码为固定大小的表示，如果有无限的资源，一个简单的方法是使用 decoder 来处理整个序列，但是在现实情况下啊是无法做到的。

为了解决这个问题 Vanilla model 将较长的语料切分多个片段去训练，但这种方式忽略了片段的所有上下文信息，这种训练模式下，信息不会在各段之间流动。

![image-20230313193253209](note_images\image-20230313193253209.png)

训练时的缺点：不能发挥出 self-attention 缓解梯度消失的优势；存在上下文分裂的情况，破坏了模型捕获长期依赖关系的能力。

评估时缺点：在评估阶段，只在一段的最后位置做一次预测，如图 1(b)，这样保证了每次预测都利用了训练过程中最长的上下文关系，同时缓解了上下文分裂情况，但是计算资源消耗大，速度很慢。

## 2 模型

### 2.1 片段回归机制 

Segment Recurrence Mechanism
$$
P(\mathbf{x})=\prod_{t} P\left(x_{t} \mid \mathbf{x}_{<t}\right)
$$
为解决上述缺点，本文提出将循环机制加入到 transformer 架构中，在训练过程中，将前一份片段计算的隐藏状态序列被 (fixed and cached) 被固定和缓存，以便模型在下一个新片段是作为拓展上下文时重用。

![image-20230313195729085](note_images\image-20230313195729085.png)



设片段长度为 $L$，两个连续的片段分别为 $s_{\tau }=[x_{\tau}, 1, ···,x_{\tau},L]$，$s_{\tau +1}=[x_{\tau+1}, 1, ···,x_{\tau+1},L]$。

$\mathbf{h}_{\tau}^n\in \mathbb{R}^{L\times d} $ 表示 第 $\tau$ 个片段 $s_{\tau}$ 产生的第 $n$ 个隐藏层状态，那么 $s_{\tau}$ 产生的第 $n$ 层隐状态是: 
$$
\begin{array}{l}
\widetilde{\mathbf{h}}_{\tau+1}^{n-1}=\left[\mathrm{SG}\left(\mathbf{h}_{\tau}^{n-1}\right) \circ \mathbf{h}_{\tau+1}^{n-1}\right], \\
\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}=\mathbf{h}_{\tau+1}^{n-1} \mathbf{W}_{q}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{k}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{v}^{\top}, \\
\mathbf{h}_{\tau+1}^{n}=\text { Transformer-Layer }\left(\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}\right) .
\end{array}
$$
其中：

$SG(h_{\tau}^{n-1})$ 函数代表 $h_{\tau}^{n-1}$ 不参与梯度计算，$[h_u \circ  h_v]$ 表示两个向量拼接。

$W_q^T, W_k^T, W_v^T$ 是模型参数。QKV 的参数。

计算 Query 是时候用的本段的前一层信息 $\mathbf{h}_{\tau+1}^{n-1}$，计算 Key 和 Value 用的是 $\widetilde{\mathbf{h}}_{\tau+1}^{n-1}$。



原则上只要 GPU 内存允许，该方法可以利用前面更多段的信息，测试阶段也可以获得更长的依赖（类似于 DenseNet）。





### 2.2 相对位置编码	

Relative Positional Encodings

提出原因：

原来的标准 Transformer 的输入是 Token_embedding + Positional_embedding，这里的位置信息是绝对位置。在切片后，如果仍然使用原来的位置的话，不同片段的同一位置具有相同的位置编码，这是不合理的，因为无法区分片段之间的关联。

因此本文提出使用相对位置编码，不再关心句中词的绝对位置信息，而是相对的，比如说两个词之间隔了多少个词这样的相对信息.通过相对位置编码，Query 向量能辨别出不同的片段的相同位置。

在标准的 Transformer 里，同一个 Segment 的 $q_i$ 和 $k_j$ 的 attention score 这样分解

$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{(d_k)}})V$ 


$$
\mathbb{} \begin{aligned}
A_{i, j}^{a b s} & = Q^{\top}K \\
& =\left(W_{q}\left(E_{x_{i}}+U_{i}\right)\right)^{\top} \cdot\left(W_{k}\left(E_{x_{j}}+U_{j}\right)\right) \\
& =\left(E_{x_{i}}+U_{i}\right)^{\top} W_{q}^{\top} W_{k}\left(E_{x_{j}}+U_{j}\right) \\
& =E_{x_{i}}^{\top} W_{q}^{\top} W_{k}\left(E_{x_{j}}+U_{j}\right)+U_{i}^{\top} W_{q}^{\top} W_{k}\left(E_{x_{j}}+U_{j}\right) \\ \\
& =\underbrace{E_{x_{i}}^{\top} W_{q}^{\top} W_{k} E_{x_{j}}}_{(a)}+\underbrace{E_{x_{i}}^{\top} W_{q}^{\top} W_{k} U_{j}}_{(b)} \\
& +\underbrace{U_{i}^{\top} W_{q}^{\top} W_{k} E_{x_{j}}}_{(c)}+\underbrace{U_{i}^{\top} W_{q}^{\top} W_{k} U_{j}}_{(d)}
\end{aligned}
$$
其中，$E_{x_i}$ 是词 $i$ 的词向量，$U_i$ 是词 $i$ 的位置向量

(a)(b)(c)(d) 四项各有各的意义：(a) 表示纯基于内容之间的寻址；(b) 和 (c) 则分别是 $i$ 位置的内容和位置信息分别相对于 $j$ 位置的位置和内容信息进行的寻址；(d) 则是纯基于位置之间的寻址。于是要改进的话，就需要对后三个和位置信息相关的项进行改进。





Transformer-XL 给出的改进方案是这样:
$$
\begin{aligned}
A_{i, j}^{r e l} & =\underbrace{E_{x_{i}}^{\top} W_{q}^{\top} W_{k, E} E_{x_{j}}}_{(a)}+\underbrace{E_{x_{i}}^{\top} W_{q}^{\top} W_{k, R} {\color[RGB]{0, 255, 255} R_{i-j}}}_{(b)} \\
& +\underbrace{{\color{Red}u^T}W_{k, E} E_{x_{j}}}_{(c)}+\underbrace{{\color{Red}v^T} W_{k, R} 	 {\color[RGB]{0, 255, 255} R_{i-j}}}_{(d)}
\end{aligned}
$$

- 和前面的 $A_{i, j}^{a b s}$ 相比，第一个改动是将 (b) 和 (d) 里的绝对位置编码 $U_{j}$ 都替换成相对位置编码向量 ${\color[RGB]{0, 255, 255}R_{i-j}}$。注意这里的 ${\color[RGB]{0, 255, 255}R}$ 是之前介绍的正弦函数的编码方式，它是固定的，不需要学习
- 在 (c) 中用可训练的 ${\color{red}u} \in R^{d}$  替代原来的  $U_{i}^{\top} W_{q}^{\top}$  。因为我们假设 Attention score 只依赖于 $i$ 和 $j$ 的相对位置，而与 $i$ 的绝对位置无关，所以这里对于所有的 $i$ 都相同。也 就是 $U^{\top} W_{q}^{\top}$，所以可以用一个新的 ${\color{red}u}$ 来表示。同理，(d) 中的 ${\color{red}v}\in R^{d}$ 也一样。
- 最后，我们把 Key 的变换矩阵 $W_{k}$ 拆分成 $W_{k, E}$ 和 $W_{k, R}$ ，分别给内容向量和相对位置向量用。



在上面的新公式里，每一项的意义都非常清晰：(a) 表示内容的计算，也就是 $x_i$ 的 Embedding 乘以变换矩阵 $W_q$ 和 $x_j$ 的 Embedding 乘以 $W_{k,E}$ 的内积；(b) 表示基于内容的位置偏置，也就是 $i$ 的向量乘以相对位置编码；(c) 表示全局的内容偏置；(d) 表示全局的位置偏置

### 2.3 结合片段回归和相对位置

N 层 Transformer-XL 结构:
$$
\begin{aligned}
\widetilde{\mathbf{h}}_{\tau}^{n-1}&=\left[\mathrm{SG}\left(\mathbf{m}_{\tau}^{n-1}\right) \circ \mathbf{h}_{\tau}^{n-1}\right], \\
\mathbf{q}_{\tau}^{n}, \mathbf{k}_{\tau}^{n}, \mathbf{v}_{\tau}^{n}&=\mathbf{h}_{\tau}^{n-1} \mathbf{W}_{q}^{n\top}, \widetilde{\mathbf{h}}_{\tau}^{n-1} \mathbf{W}_{k}^{n\top}, \widetilde{\mathbf{h}}_{\tau}^{n-1} \mathbf{W}_{v}^{n\top}, \\
\mathbf{A}_{\tau,i,j}^{n}&= \mathbf{q}_{\tau,i}^{n\top} \mathbf{k}_{\tau,j}^{n}+ \mathbf{q}_{\tau,i}^{n\top} \mathbf{W}_{k,R}^{n}\mathbf{R}_{i-j}+u^{\top} \mathbf{k}_{\tau,j}+v^{\top}\mathbf{W}_{k,R}^n\mathbf{R}_{i-j}   \\
\mathbf{a}_{\tau}^{n} &= \text {Masked-Softmax}(\mathbf{A}_{\tau}^n) \mathbf{v}_{\tau}^{n} \\
\mathbf{o}_{\tau}^{n} &= \text{LayerNorm(Linear}(\mathbf{a}_{\tau}^n) + \mathbf{h}_{\tau}^{n-1} ) \\
\mathbf{h}_{\tau+1}^{n}&=\text {Positionwise-Feed-Forward}\left(\mathbf{o}_{\tau}^{n}\right)
\end{aligned}
$$
$\mathbf{h}_{\tau}^0 := \mathbf{E}_{\mathbf{s}_{\tau}}$ 表示 word embedding





## 3 实验

评价指标

1. bpc (bits-per-character) 
2. PPL (Perplexity) 

都是越小越好。

当计算基于字符长度单位的混淆度 (Perplexity)时，$Perplexity = 2^{bpc}$.



一种评估序列模型有效上下文长度( ECL )的方法。ECL是指增加上下文跨度将导致增益超过阈值的最长长度。然而，ECL忽略了一个事实，即当一个模型已经达到较低的困惑度时，仅仅使用较短的上下文就很难得到改善，因此不适合多个模型之间的公平比较。



本文提出 RECL (Relative Effective Context Length)

RECL 基于一组模型，每组有同样的参数量，相同的 baseline



相比 vanilla Transformer，Transformer-XL 快了1874倍。







参考：

Transformer-XL https://wmathor.com/index.php/archives/1475/

评价指标 https://blog.csdn.net/weixin_43922901/article/details/103218081
