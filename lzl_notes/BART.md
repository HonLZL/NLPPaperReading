BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension



## 1 背景

BERT 是 transformer 的 encoder 的双向形式，缺少 tokens，不利于生成；GPT 是 transformer 的decoder部分是有顺序的，不利于学习句子的双向信息。

BART 可以看作是 BERT 和 GPT 的结合，似乎又回到了 transformer。

预训练分为两个阶段：

( 1 )文本被任意噪声函数破坏

( 2 )学习一个序列到序列的模型来重建原始文本。



## 2 模型

采用 transformer 

![image-20230215163054801](note_images\image-20230215163054801.png)

双向 Encoder + Decoder



<img src="note_images\image-20230215194738400.png" alt="image-20230215194738400" style="zoom:50%;" />



Token Masking：随机 mask 一些字，Bert 就是这样预训练的

Sentence Permutation: 打乱句子排列

Document Rotation: 反转句子，让模型学习辨别开头

Token Deletion: 随机删除一些 token

Text Infilling: 对多个文本片段，每个片段的词有概率被 mask 代替，0-length ，片段的长度是 $\lambda=3$ 的泊松分布。该方法让模型学习预测 一个片段有多少个 token 被 mask



token 是 一个句子分解为以 字词 为单位的数据结构，这个过程是分词（tokenization），分词后的 字 或者 词 就是一个 token。

span 表示一个句子的一个片段。



## 3 BART 下游任务的微调

Sequence Classification Tasks 序列分类

编码器和解码器的输入相同，解码器最终的隐藏状态被喂进多分类层



Sequence Generation Tasks 序列生成

由于自回归的 Decoder 



Token Classification    Tasks token 分类

使用 解码器 的顶层隐藏层作为每个单词的表示



Machine Translation   机器翻译

作者采用新的随机初始化 Encoder 替换 BART 编码器的 Embedding  层。该模型以端到端的方式进行训练，即训练一个新的编码器将外来词映射到输入。新的编码器可以使用不同于原始 BART 模型的词汇。其中随机初始化  Encoder 的训练分两步，均需要将来自 BART 模型输出的交叉熵损失进行反向传播。第一步，作者冻结 BART  的大部分参数，仅更新随机初始化的 Encoder、BART 位置嵌入和 BART  编码器第一层的自注意力输入投影矩阵。第二步，作者将所有模型参数进行少量迭代训练



![image-20230215204211828](note_images\image-20230215204211828.png)



## 4 Results

![image-20230215205753424](note_images\image-20230215205753424.png)



用 Document Rotation 和 Sentence Shuffling 效果都比较差。Text Infilling 表现很好



评价指标

PPL(perplexity)：困惑度，越小越好。根据每个词来估计一句话出现的概率，并用句子长度作normalize。
$$
PPL(S) = P(w_1w_2···w_N)^{-\frac{1}{N}} \\
=\sqrt[N]{\frac{1}{p(w_1w_2···w_N)}} \\
=\sqrt[N]{\prod_{i=1}^{N}\frac{1}{p(w_1|w_2···w_{i-1})}}
$$
S   当前句子；
N   句子长度；
$p(w_i)$  第i个词的概率
$p(w_i|w_1w_2w_3…w_{i-1})$  这个表示基于前 i-1 个词，计算得出第 i 个词的概率

由公式可以看出，$p(w_i)$ 越大，ppl 就越小，每个词出现的概率就越高，效果就越好。

