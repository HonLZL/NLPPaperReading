# XLNet: Generalized Autoregressive Pretraining for Language Understanding

## 1 背景

Bert 依靠 mask 预训练，忽略了被 mask 位置的依赖关系，导致训练和预测是有差异的。

XL 将自回归融入进预训练



自回归  Auto Aggressive 和 自编码  Auto Encoder

<img src="note_images\image-20230222142635545.png" alt="image-20230222142635545" style="zoom: 67%;" />

GPT 是典型的自回归模型，缺点是忽略了上下文信息。Bert 是自编码模型，

XLNet 想融合 AR 和 AE 的优点。

### 2 Permutation Language Model 排列语言模型

XLNet 通过 Permutation 将 AR 和 AE 融合起来。

对一个长度为 T 的序列，那么这个序列将会有 T！个排列方式，从直觉上来说，如果模型的参数在所有的排列共享，那么模型将能够学习到双向的信息。为验证这个想法: 
$$
\underset{\theta}{max} \ \ \  \mathbb{E}_{\textbf{z} \sim Z_T} \left [\sum_{t=1}^T log p_{\theta}(x_{z_t}|x_{\textbf{z}_{<t}})\right ]
$$
$Z_T$ 表示序列的所有排列组成的集合，$z \in Z_T$ , z 表示 其中一种排列。

$z_t$ 表示排列的第几个元素，$z_{<t}$ 表示排列 $z$ 的第 1 到第 $t-1$ 个元素。

$\theta$ 表示模型参数，$p_{\theta}$ 表示似然概率。

该公式，从所有排列中选取一个排列 $z$ ，计算条件概率并求和。参数 $\theta$ 在所有序列共享，因此 $x_t$ 看到了每个其他的元素。

由于排列太多，所以通常只随机采用部分序列。

论文中，并不是采用随机打乱序列的方式来实现 Permutation，而是使用原始序列的编码，依靠 transformer 的注意力掩码来实现分解顺序的置换，因为在微调时，只会遇到自然顺序的文本。

<img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230222170753934.png" alt="image-20230222170753934" style="zoom:67%;" />



例如原序列为 1234，随机抽取一种排列为 3241，根据这个排列进行 Attention Mask。

第一行：根据自回归的原理，1 可以看到 3 2 4，所以对应的圆圈标红。

第二行：2 只能看到 3，所以只有第 3 个圈标红。

第三行：3 啥都不能看到，因此没有标红。

第四行：4 可以看到 3 和 2，所以第三个和第二个标红。

标红的意思是没被遮盖，白色代表被遮盖。





























给定相同输入序列 x，在不同分解顺序下预测 $x_3$ 的例子。

<img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230222154440166.png" alt="image-20230222154440166" style="zoom:67%;" />







参考： https://wmathor.com/index.php/archives/1475/