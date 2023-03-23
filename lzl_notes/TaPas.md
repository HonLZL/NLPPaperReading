# TAPAS: Weakly Supervised Table Parsing via Pre-training

## 1 背景

目标：从半结构化的表格中进行问答。



   传统的基于表格的自然语言问答通常被建模为语义解析（semantic parsers）任务，语义解析依赖于有监督的训练数据，这些数据与逻辑形式匹配，可以针对表执行以检索正确的表示，但是这样的数据标注很贵。

​	为了减轻收集自然语言的逻辑形式表达的庞大开销，有采取改写，human in the loop（人工参与训练），训练其他领域的数据。一个流行的数据收集方法侧重于弱监督，训练一个问题和它的外延(denotation)，而不是逻辑形式。但是由于错误的逻辑形式和奖励稀疏，从该输入中训练一个语义解析是困难的。

​	语义解析仅使用生成的逻辑形式作为检索答案的中间步骤。然而生成逻辑形式会引入困难，例如如何保持有足够表达能力的逻辑形式，服从解码约束，和标签偏差问题。

​	为解决上述问题，本文提出 TaPas，不用生成逻辑形式的弱监督问答模型。通过选择表格的子集，然后对其进行聚合操作，来预测最简方案，因此 TaPas 能够从自然语言中学习操作，而不需要在某种形式上指定它们，这是通过扩展 BERT 架构来实现的。



## 2 模型

### 2.1 Embedding

输入的 embedding 有 66 个，分别是 Token，Position，Segment，Column，Row，Rank。

前面是 question，后面是 table 的摊开形式，herder，然后一行接一行的格式。如下图：

![image-20230321190730575](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230321190730575.png)



> **Position ID**
>
> ​	token 在展平的序列中的**索引**
>
> **Segment ID**
>
> ​	0 代表**问题**，1 代表 **表头** 和 **单元格**
>
> **Column / Row ID**
>
> ​	token 出现的 行或列的 索引，就是第几行第几列。0 代表这个 token 出现在 **问题** 中
>
> **Rank ID**
>
> ​	如果列的值能被解析为 小数 或者 是日期，就对它进行排序，并且根据数值 rank 分配一个 embedding，0 表示不具有可比性，1 表示最小项，$i+1$ 表示第 $i$ 项。这样能帮助模型处理最高级问题，因为 word pieces 在信息上也许不代表数字。
>
> **Previous Answer**
>
> ​	`Previous Answer` 给出了一个会话的设置，当前的问题可能是前一个问题或答案。例如连续问两个问题，第二个问题是根据第一个问题的答案接着问的。

### 2.2 模型结构

TaPas 模型基于带有 **additional positional** embeddings BERT 的 encoder 去 编码 **表格结构数据**。

token的输入是 `问题`、 `table` 和 2.1 的额外的 embedding，需要将 table 变成一维，所以有个 flatten 操作，就是把二维表格一行一行在后面加上。

两个分类层，用于选择单元格和选择作用于单元格的聚合操作。

模型的最后一层的[CLS]拿出来做aggregation prediction(SUM/COUNT/AVG/NONE)，linearized table拿来预测选择哪些单元格。

例如 问题是 `Total number of days for the top two？` 

右上角的表格显示，模型选择 第一行和第二行，左上角显示，聚合函数为 SUM。

![image-20230321184217453](note_images\image-20230321184217453.png)

#### 2.2.1 单元格选择

目的是选择表格的子集，这些单元格能作为答案或者聚合函数的输入。

单元格被建模为独立的伯努利变量

1. 在 token 的最后一个隐藏向量上使用一个线性层来计算它的 logit
2. 单元格 logits 为该单元格中的 token 的平均 logits
3. 该层的输出作为选择该单元格的概率 $p_s^{(c)}$，$c$ 表示该单元格。



此外，我们发现添加**归纳偏差**以选择单个列中的单元格是有用的，通过引入分类变量去选择正确队列来实现这个。

该模型通过对给定列中出现的单元格的平均 embedding 应用新的线性层来计算该列的logit。

还添加一个额外的列 logit ，对应于不选择列或单元格。

该层的输出是选择列 $co$ 的概率 $p_{col}^{(co)}$，是对列的 logits 应用 softmax 函数计算出的。



#### 2.2.2 聚合函数选择

在问答时，有可能需要用到聚合函数。SUM/COUNT/AVG/NONE

为了不产生逻辑形式，TaPas 输出单元格的子集和可以选择的聚合函数集合。

聚合操作符由一个 线性层选择，然后在第一个标记([CLS])的最后的隐藏向量上做一个softmax。

我们将这一层表示为$p_a^{(op)}$，其中 $op$ 是某个聚合运算符。



#### 2.2.3 推理

上面两步预测出了最有可能的单元格和聚合函数，

为了预测离散的单元格选择，我们选择所有概率大于 0.5 的单元格，然后通过聚合函数来检索答案。



## 3 预训练

在维基百科的大量表格预训练模型，这样模型能够学习到文本和表之间，以及列的单元格和他们的标题之间的相关性。

预训练的数据来源与维基百科上的大量文本表格对，作者提取了 3.3M 的 Infobox 和 2.9M 的 WikiTable ，所谓 Infobox 如下图右侧所示：

<img src="note_images\image-20230322163547783.png" alt="image-20230322163547783" style="zoom: 50%;" />

最多考虑有 500 个单元格的表。

使用 <th> 标签去识别表头。

使用其中的水平级别的表格，就是只有一个header row多个columns名字的那种普通表格。作者把  Infobox 转换成只有一列的数据。

至于对应的问题(query)，作者提取了表格相对应的表格标题、文章标题、文章的简介、文章的小标题和分隔表格的文字。



将提取的 text-table 对作为预训练:

1 和 Bert 一样，使用 MLM 目标

2 另外添加第二个目标：这个表是否属于这段文本，或者是随即表，但没有发现这可以提高最终任务的性能。





## 4 微调

![image-20230322170741880](note_images\image-20230322170741880.png)



​	首先给出训练集合，集合中由 $N$ 个实例组成，表示为 $\{(x_i,T_i,y_i)\}_{i=1}^N$ 也就是〈question, cell coordinates, scalar answer〉。$x_i$ 代表语句，$T_i$ 是一个表，$y_i$ 是对应一个结果。目标是学习一个模型，能将 utterance $x$ 和 program $z$ 对应起来。例如当 $z$ 对表 $T$ 执行时，程序能够给出正确的 $y$。$z$ 就是一个表的所有单元格的子集和一个聚合操作。

​	将每个实例的表示集合 $y$ 转化为 $(C,s)$ ，$C$ 是多个单元格，$s$ 是一个标量，当且仅当 $y$ 是单个标量时才填充 $s$。对于没有填充的 $s$，通过训练模型来选择 $C$ 中的单元格。

​	对于标量答案，$s$ 被填充，但 $C$ 是空的，我们训练模型去预测 等于 $s$ 的表格单元格 上的聚合操作。

### 4.1 Cell selection

$y$ 被映射到表的多个单元格 $C$ 的子集。也就是说，答案就是某个单元格的内容。

对应 Question 1，直接拿单元格的内容进行输出。

分层模型：首先选择列，再只从该列中选择出单元格。

直接训练模型去选择 在 $C$ 中有最高概率单元格的 列。因为单元格一定在单个列中，这种对模型的限制提供了有用的归纳偏差。如果 $C$ 是空的，我们选择额外的空列对应于空单元格选择。

最后模型去训练选择 $C\cap col$ ，不选择 $(T\setminus C)\cap col$，

==总损失由以下三部分组成==

>  **1 列损失 ** 
> $$
> \mathcal{J}_{\text {columns }}=\frac{1}{\mid \text { Columns } \mid} \sum_{\text {co } \in \text { Columns }} \mathrm{CE}\left(p_{\mathrm{col}}^{(\mathrm{co})}, \mathbb{1}_{\mathrm{co}=\mathrm{col}}\right)
> $$
> ​      列的集合包括额外的空列。$CE(·)$ 是交叉熵损失，$\mathbb{1}$ 是指示函数，为真是取 1，否则取 0
>
> **2 单元格损失 **
> $$
> \mathcal{J}_{\text {cells }}=\frac{1}{|\operatorname{Cells}(\mathrm{col})|} \sum_{c \in \operatorname{Cells}(\mathrm{col})} \operatorname{CE}\left(p_{\mathrm{s}}^{(c)}, \mathbb{1}_{c \in C}\right)
> $$
> ​      $\text {Cells(col)}$ 是被选中列的单元格的集合。
>
> **3 聚合函数损失** 
> $$
> \mathcal{J}_{\text {aggr }}=-\log p_{\mathrm{a}}\left(o p_{0}\right)
> $$
> 对于没有聚合操作的情况，聚合设置为 NONE，赋值为 $op_0$，
>
> **总损失  **
> $$
> \mathcal{J}_{\text {aggr }}=\mathcal{J}_{\text {columns }}+\mathcal{J}_{\text {cells }}+\alpha\mathcal{J}_{\text {aggr }}
> $$
> ​      $\alpha$ 是放缩超参数



### 4.2 Scalar answer

结果 $y$ 是一个标量 $s$ ，并且这个标量没有在表中显示，也就是没有对应的单元格，即 $C=\emptyset$。例如 Question 2。

这个通常对应于多个列的聚合操作，COUNT, AVERAGE and SUM

$op$ 聚合函数，$p_s$ 选则一个单元格的概率，$T$ 是表格。

具体计算有以下三个：

> $op$        compute($op$, $p_s$, $T$) 
>
> ------
>
> COUNT       $\sum_{c\in T}p_s^{(c)}$             选择单元格c的概率的和
>
> SUM         $\sum_{c\in T}p_s^{(c)} · T[c]$       选择的每个单元格c的概率与对应单元格中的值的乘积的和。
>
> AVERAGE     $\frac{compute(SUM,p_s,T)}{compute(COUNT,p_s,T)}$    用上面的SUM除以COUNT

最终的计算公式预测值为每个操作的概率乘以对应的compute值：
$$
s_{\mathrm{pred}}=\sum_{i=1} \hat{p}_{\mathrm{a}}\left(o p_{i}\right) \cdot \operatorname{compute}\left(o p_{i}, p_{\mathrm{s}}, T\right)
$$
其中 $ \hat{p}_{\mathrm{a}}\left(o p_{i}\right)=\frac{ p_{\mathrm{a}}\left(o p_{i}\right)}{ \sum_{i=1} p_{\mathrm{a}}\left(o p_{i}\right)}$ ，是除了 NONE 的聚合操作的概率。

损失为:
$$
\mathcal{J}_{\text {scalar }}=
\left\{\begin{matrix} 
  0.5\cdot a^2 \ \ \ \ \ \ \  a\le \delta \\  
  \delta\cdot a-0.5 \cdot \delta^2 \ \ \ \ \ \text{otherwise }
\end{matrix}\right.
$$
其中 $a=|s_{pred}-s|$ ，$s$ 是 ground truth ，$\delta$ 是超参数。

Scalar answer 也蕴含着聚合操作，因此也需要聚合损失去惩罚给 NONE 分配概率的模型：
$$
\mathcal{J}_{\text {aggr }}=-\log p_{\mathrm{a}}\left(o p_{0}\right)
$$
总的损失为 $\mathcal{J}_{\text {SA}}=\mathcal{J}_{\text {aggr }}+\beta \mathcal{J}_{\text {scalar}}$ 。

对于一些实例，$\mathcal{J}_{\text {scalar}}$ 可能很大，因此引入 $cutoff$ 参数，当 $\mathcal{J}_{\text {scalar}}>cutoff$，就将 $\mathcal{J}=0$ 去完全忽略这个实例。

计算（computation）在训练过程是连续的，在推理过程是离散的，我们进一步添加了一个 temperature，这个 temperature 对 token logits 进行缩放，使得 $p_s$ 的输出更接近二进制值。



### 4.3 Ambiguous answer

一个标量答案 $s$ 出现在表格中（因此 $C\ne \emptyset$ ），但实际正确的含义并不是单元格的内容，例如（Question 3），正确答案是2，表格里也有 2，但是答案 2 和单元格里的 2 意义是不一样，答案的 2 需要聚合操作的参与。在（Question 4）中，答案是 7 ，单元格也有 7 ，这里的答案 7 和 单元格的 7 意义是相同的。由于答案是否就是单元格的内容还是通过聚合操作得出的结果是 ambiguous 。

在这种情况下，我们根据下面的规则动态地让模型选择 cell selection 或 scalar answer。
$$
ambiguous\ answer=
\left\{\begin{matrix} 
  cell\ section  \ \ \ \ \ \ p_a(op_0)\ge S \\  
  sclar\ answer  \ \ \ \ \ otherwise
\end{matrix}\right.
$$
其中 $0 < S < 1$ ，$S$ 是一个阈值参数。



## 5 总结

TaPas 将单个的表格作为上下文处理，所以

无法处理非常大的表格

无法处理多个表格联合

无法处理多重聚合的问题，



















