# TableFormer: Robust Transformer Modeling for Table-Text Encoding

## 1 背景

以 TaPas 为代表的现有模型容易受到扰动，缺乏结构偏差，单元格联系较差。

> ​	为了在序列化的表格中加入表格结构信息，在编码单元格时，TAPAS使用了**行 ID 向量和列 ID 向量作为额外的特征**，其他很多模型把表头中相应的列名当作额外的特征或者单元格的前置 token。
>
> ​	然而，行列 ID 以及 BERT（BART）中存在的序列相对位置或者绝对位置编码会**引入表格行列顺序的虚假偏置**（spurious bias）。在回答绝大多数问题时，我们希望模型真正理解表格内容，而不是根据行列顺序的虚假偏置来作出判断。理想情况下，模型只需要知道同行同列信息即可，不需要知道额外的行及列的顺序信息。
>
> ​	实验表明，用TAPAS执行表格问答任务时，如果在预测阶段加入随机的行列扰动，模型的表现会下降4%-6%。



例如：

| Song Title   | Length |
| ------------ | ------ |
| Screwed Up   | 5:02   |
| Ghetto Queen | 5:00   |

Question: Of all song lengths, which one is the longest? 

Gold Answer: 5:02 

TAPAS: 5:00 

TAPAS after row order perturbation: 5:02 ，换了个顺序，结果又是正确的了

TABLEFORMER: 5:02

这个例子表明，TaPas 受到顺序的影响

由标题可以看出， TableFormer 是一个具有鲁棒性的模型。



## 2 模型

### 2.1 模型输入

1. Token Embedding 
2. Segment Embedding
3. Rank Embedding
4. 与 TaPas 相比没有 Column/Row ID，我们不需要单元格的顺序
5. Per cell positional，TaPas 是 token 在展平的序列中的**索引**，而 TableFormer 是每个单元格独立的位置，就算每个单元格的 tokens 都是从 0 开始编码。

​    删除这些行列顺序相关的信息后，表格结构的编码完全由`行列注意力偏置`实现。这样，**无论如何扰动表格行列顺序，在 TableFormer 的视角，输入总是完全相同的**，由此可以保证扰动前后预测的完全一致性。



### 2.2

==**Positional Encoding in TABLEFORMER**== 

>  我们需要的位置信息是两个单元格是否在同一列或同一行,和是否在单元格的表头，而不是包含他们的行和列的绝对顺序。
>
>  函数 $\phi(v_i,v_j):\ V\times V \to \mathbb{N}  $ ，代表着 $v_i$ 和 $v_j$ 在序列中的关系，$\phi$ 代表 在 table-text对中，任意 tokens 之间的关系。



==**Attention Biases in TABLEFORMER**== 

> 为了解决上述问题，并且进行更好的表格结构编码以及文本表格内容对齐，本文**引入了13种结构化注意力偏置**，每种颜色代表一种偏置。每种偏置用一个可学习的标量表示，与 self-attention 中key value 计算好并且 scaling 之后的 similarity score 相加，以**针对不同的结构位置自动调整注意力程度**。
> $$
> &\hat{A}_{ij} = b_{\phi(v_i,v_j)} \\
> &A_{ij}=\frac{(h_i^\top W^Q)(h_i^\top W^K)^\top}{\sqrt{d_K}}+\hat{A}_{ij}\\
> &Attn(H)=softmax(A)V
> $$
> 
>
> <img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230330150010779.png" alt="image-20230330150010779" style="zoom:50%;" />
>
> `same row` 有助于识别相同行信息
>
> `same column`, `header to column cell`, `cell to column header` 融合了一样的列信息，没有列的顺序信息。
>
> `cell to column header` 使每个单元格都知道自己的列标题
>
> `header to sentence`, `cell to sectence` 有助于对齐文本的 column grounding(列内容) 和 cell grounding
>
> `sentence to header`, `sentence to cell`, `sentence to sentence` 有助于以表格为上下文，去理解句子
>
> `header to same header`, `header to other header` 有助于更好地理解表的格式，
>
> `same cell bias` 有助于更好地理解单元格内容

 删除这些行列顺序相关的信息后，表格结构的编码完全由行列注意力偏置实现。这样，**无论如何扰动表格行列顺序，在 TableFormer 的视角，输入总是完全相同的**，由此可以保证扰动前后预测的完全一致性。



![image-20230330125719861](note_images\image-20230330125719861.png)



## 3 实验

本文在三个表格理解与推理的数据集上进行了实验，分别为SQA（基于表格的连续问答），WTQ（基于表格的复杂问答），和TabFact（基于表格的事实核查）。

除了标准的评测数据，本文还将SQA和TabFact的测试集中的表格行列施加随机扰动，构造了扰动测评的场景。



除了标准的评测指标，本文还提出了一个 **VP（variation percentage）指标**来度量所有测试样例中扰动前后预测出现变化的比例的下界，可以作为样例水平鲁棒性的上界（VP越低越鲁棒）。(排列前后结果不同的样本 / 所有样本)
$$
VP=\frac{(t2f+f2t)}{(t2t+t2f+f2t+f2f)}
$$


  另一个很自然的想法来解决表格行列顺序扰动带来的影响是在训练时随机扰动表格进行数据扩增，而不改变TAPAS的模型结构。本文也对TableFormer和这种数据扩增进行了对比实验。实验表明，数据扩增虽然可以减小扰动对于模型总体效果的影响，但最好的表现依然逊色于TableFormer，可能由于TableFormer带来了额外的有效偏置信息（如文本表格对齐等）。

并且，数据扩增无法保证样例水平模型预测对扰动的鲁棒性，从而VP远远高于TableFormer接近于0的VP
