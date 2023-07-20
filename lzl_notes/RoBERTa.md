# RoBERTa: A Robustly Optimized BERT Pretraining Approach

## 1 背景

是 BERT 的强化版本

用更大的模型参数量。用更大的 batch size。用更多的训练数据。



## 2 优化的方法

### 2.1 动态掩码： Dynamic Masking

**原始静态掩码：**

准备训练数据时，每个样本只会进行一次随机mask（因此每个epoch都是重复），后续的每个训练步都采用相同的mask，这是**原始静态mask**，即单个静态mask，这是原始 BERT 的做法。

**修改版静态掩码：**

在预处理的时候将数据集拷贝 10 次，每次拷贝采用不同的 mask（总共40 epochs，所以每一个mask对应的数据被训练4个epoch）。这等价于原始的数据集采用10种静态 mask 来训练 40 个 epoch。

**动态MASK：**

并没有在预处理的时候执行 mask，而是在每次向模型提供输入时动态生成 mask，所以是时刻变化的。在更大的数据集中，有效性将会更能体现。

| Masking               | SQuAD 2.0 | MNLI-m | SST-2 |
| --------------------- | --------- | ------ | ----- |
| 原始 static reference | 76.3      | 84.3   | 92.8  |
| 修改 static           | 78.3      | 84.3   | 92.5  |
| dynamic               | 78.7      | 84.0   | 92.9  |



### 2.2 是否使用 NSP

为了探索NSP训练策略对模型结果的影响，将一下4种训练方式及进行对比：

**SEGMENT-PAIR + NSP：**
这是原始 BERT 的做法。输入包含两部分，每个部分是来自同一文档或者不同文档的 segment （segment 是连续的多个句子），这两个segment 的token总数少于 512 。预训练包含 MLM 任务和 NSP 任务。



**SENTENCE-PAIR + NSP：**
输入也是包含两部分，每个部分是来自同一个文档或者不同文档的单个句子，这两个句子的token 总数少于 512。由于这些输入明显少于512 个tokens，因此增加batch size的大小，以使 tokens  总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务。



**FULL-SENTENCES：**
输入只有一部分（而不是两部分），来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512 。输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token 。预训练不包含 NSP 任务。



**DOC-SENTENCES：**
输入只有一部分（而不是两部分），输入的构造类似于FULL-SENTENCES，只是不需要跨越文档边界，其输入来自同一个文档的连续句子，token 总数不超过 512 。在文档末尾附近采样的输入可以短于 512个tokens， 因此在这些情况下动态增加batch size大小以达到与  FULL-SENTENCES 相同的tokens总数。预训练不包含 NSP 任务。



<img src="note_images\image-20230510205223479.png" alt="image-20230510205223479" style="zoom: 67%;" />

但是 **DOC-SENTENCES** 的格式导致了多变的 batch sizes，所以本文采用 **FULL-SENTENCES**。



### 2.3 更大的 batch size

​	在以往的经验中，当学习速率适当提高时，采用非常大 mini-batches 的训练既可以提高优化速度，又可以提高最终任务性能。但是论文中通过实验，证明了更大的batches可以得到更好的结果，实验结果下表所示。

![image-20230510205754550](note_images\image-20230510205754550.png)

因为更大的 batch size 更容易并行，因此本文采用 8k 的 batch size。



### 2.4 Text 编码

两种BPE实现方式：

- 基于 char-level ：原始 BERT 的方式，它通过对输入文本进行启发式的词干化之后处理得到。
- 基于 bytes-level：与 char-level 的区别在于bytes-level 使用 bytes 而不是 unicode 字符作为 sub-word 的基本单位，因此可以编码任何输入文本而不会引入 UNKOWN 标记。

原始 Bert 使用 character-level BPE，本文采用更大的 byte-level BPE。当采用 bytes-level 的 BPE 之后，词表大小从3万（原始 BERT 的 char-level ）增加到5万。这分别为  BERT-base 和  BERT-large 增加了1500万和2000万额外的参数。之前有研究表明，这样的做法在有些下游任务上会导致轻微的性能下降。但是本文作者相信：这种统一编码的优势会超过性能的轻微下降，且作者在未来工作中将进一步对比不同的encoding方案。



### 2.5 数据 和 steps

从原来的 16G，提升到 160G 数据集。

增加训练 steps 到 500k，作者观察到训练时间最长的模型，也不会过拟合数据，并且会从额外的训练中收益。

<img src="note_images\image-20230511143711891.png" alt="image-20230511143711891" style="zoom:50%;" />



## 3 结果

**R**obustly **o**ptimized **BERT** **a**pproach

结果很好，实现了 state-of-the-art results on GLUE, RACE and SQuAD



















