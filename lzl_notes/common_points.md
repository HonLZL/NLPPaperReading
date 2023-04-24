## 0 词向量

### 1 Word2vec

方法假设：文本中离得越近的词语相似度越高。

主要用 skip-gram 计算

![image-20221125150915702](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221125150915702.png)



> **评估 word2vec 效果** 
>
> 1. 输出与特定词语相关度比较大的词语
> 2. 可视化
> 3. 类比实验：国王-王后=男人-女人
>
> **缺点** 
>
> 1. 没有考虑多义词
> 2. skip-gram时窗口长度有限
> 3. 没有考虑全局文本信息
> 4. 不是严格意义的语序
> 5. ······



### 2 glove

[详解](http://www.fanyeong.com/2018/02/19/glove-in-detail/) 



## 1 BP神经网络

Back-Propagation 反向传播

[知乎](https://zhuanlan.zhihu.com/p/486303925) 



## 2 MLP（多层感知机）



[MLP](https://aistudio.csdn.net/62e38aaecd38997446774dcb.html?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-1-93405572-blog-102802517.pc_relevant_recovery_v2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-1-93405572-blog-102802517.pc_relevant_recovery_v2&utm_relevant_index=1) 



## 3 LSTM(长短期记忆网络)

[B站](https://www.bilibili.com/video/BV1JU4y1H7PC/?spm_id_from=333.337.search-card.all.click&vd_source=2ca6993c5ecbc492902a1449c800fe3d) 

[视频](https://www.bilibili.com/video/BV1Z34y1k7mc/?share_source=copy_web&vd_source=2ca6993c5ecbc492902a1449c800fe3d) 





## 4 





## 5 CRNN





## 6 自定义学习率动态调整



## 7 backbone, head, neck, botteneck

### 7.1 backbone

主干网络，大多时候指的是提取特征的网络，其作用就是提取图片中的信息，供后面的网络使用。

常用 backbone 有 resnet，VGG，而不是我们自己设计的网络，因为这些网络已经证明了在分类等问题上的特征提取能力是很强的。在用这些网络作为backbone的时候，都是直接加载官方已经训练好的模型参数，后面接着我们自己的网络。让网络的这两个部分同时进行训练，因为加载的 backbone 模型已经具有提取特征的能力了，在我们的训练过程中，只需进行**微调**，使得其更适合于我们自己的任务。



### 7.2 head

head是获取网络输出内容的网络，利用之前提取的特征，head利用这些特征，做出预测。



### 7.3 neck

是放在backbone和head之间的，是为了更好的利用backbone提取的特征

 

### 7.4 bottleneck

瓶颈的意思，通常指的是网络输入的数据维度和输出的维度不同，输出的维度比输入的小了许多，就像脖子一样，变细了。经常设置的参数 bottle_num=256，指的是网络输出的数据的维度是256 ，可是输入进来的可能是1024维度的。



## 8 各种分词

### 8.1 为什么要分词

​    模型是无法识别原始文本的，需要转换成数字的形式，一个数字代表哪个文字，需要由分词处理。模型的整个过程是：输入句子-->预处理，分词--->变成id或数字后进入模型-->输出结果。

tokenize的目标是把输入的文本流，**切分成一个个子串，每个子串相对有完整的语义**，便于学习embedding表达和后续模型的使用。

tokenize有三种粒度：**word/subword/char**

- **word/词**，词，是最自然的语言单元。对于英文等自然语言来说，存在着天然的分隔符，如空格或一些标点符号等，对词的切分相对容易。但是对于一些东亚文字包括中文来说，就需要某种分词算法才行。顺便说一下，Tokenizers库中，基于规则切分部分，**采用了spaCy和Moses两个库**。如果基于词来做词汇表，由于长尾现象的存在，**这个词汇表可能会超大**。像Transformer XL库就用到了一个**26.7万**个单词的词汇表。这需要极大的embedding matrix才能存得下。embedding matrix是用于查找取用token的embedding vector的。这对于内存或者显存都是极大的挑战。常规的词汇表，**一般大小不超过5万**。
- **char/字符**，即最基本的字符，如英语中的'a','b','c'或中文中的'你'，'我'，'他'等。而一般来讲，字符的数量是**少量有限**的。这样做的问题是，由于字符数量太小，我们在为每个字符学习嵌入向量的时候，每个向量就容纳了太多的语义在内，学习起来非常困难。
- **subword/子词级**，它介于字符和单词之间。比如说'Transformers'可能会被分成'Transform'和'ers'两个部分。这个方案**平衡了词汇量和语义独立性**，是相对较优的方案。它的处理原则是，**常用词应该保持原状，生僻词应该拆分成子词以共享token压缩空间**。



传统空格分隔tokenization技术的问题

- 传统词表示方法无法很好的处理OOV问题，对于一些罕见词，直接就会变成<UNK>。
- 词表中的低频词/稀疏词在模型训练过程中无法得到充分训练，进而模型不能充分理解这些词的语义
- 一个单词因为不同的形态会产生不同的词，如由"look"衍生出的"looks", "looking", "looked"，显然这些词具有相近的意思，但是在词表中这些词会被当作不同的词处理，一方面增加了训练冗余，另一方面也造成了大词汇量问题。
- 词表可能非常之大。尤其是假如要训多种语言(例如mBERT)，不同语言的词语实在太多，根本无法存下。

**OOV问题** 

> 就是 Out-Of-Vocabulary
>
> 数据集的某些词没有没有在字库里。subword 可以解决

以下三个全是 subword 分词

### 8.2 Byte-Pair Encoding (BPE)

==常规 BPE== 

BPE，即字节对编码。其核心思想在于将**最常出现的子词对合并，直到词汇表达到预定的大小时停止**。

- 首先，它依赖于一种预分词器pretokenizer来完成初步的切分。pretokenizer可以是简单基于空格的，也可以是基于规则的；
- 分词之后，统计每个词出现的频次，供后续计算使用。例如，我们统计到了5个词的词频

> ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

- 建立基础词汇表，包括所有的字符，即：

> ["b", "g", "h", "n", "p", "s", "u"]

- 根据规则，我们分别考察2-gram，3-gram的基本字符组合，**把*高频*的n-gram组合依次加入到词汇表中**，直到词汇表达到预定大小停止。比如，我们计算出ug/un/hug三种组合出现频次分别为20，16和15，加入到词汇表中。
- 最终词汇表的大小= 基础字符词汇表大小 + 合并串的数量，比如像GPT，它的词汇表大小 40478 = 478(基础字符) + 40000（merges）。添加完后，我们词汇表变成：

> ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]

实际使用中，如果遇到未知字符用<unk>代表。

==Byte-level BPE== 

BPE的一个问题是，如果遇到了unicode，**基本字符集可能会很大**。一种处理方法是我们**以一个字节为一种“字符”**，不管实际字符集用了几个字节来表示一个字符。这样的话，基础字符集的大小就锁定在了**256**。

例如，像**GPT-2**的词汇表大小为50257 = 256 + <EOS> + 50000 mergers，<EOS>是句子结尾的特殊标记。

### 8.3 WordPiece

WordPiece，从名字好理解，它是一种 **子词粒度的tokenize算法** subword tokenization algorithm，很多著名的Transformers模型，比如BERT/DistilBERT/Electra都使用了它。

它的原理非常接近BPE，不同之处在于它做合并时，并不是直接找最*高频*的组合，而是找能够**最大化训练数据似然的merge**。

如何选择两个subword进行合并：BPE选择频数最高的相邻subword合并，而WordPiece选择能够提升语言模型概率最大的相邻subword加入词表。



### 8.4 Unigram Language Model

ULM是另外一种subword分隔算法，它能够输出带概率的多个子词分段。它引入了一个假设：所有subword的出现都是独立的，并且subword序列由subword出现概率的乘积产生。WordPiece和ULM都利用语言模型建立subword词表。

与BPE或者WordPiece不同，Unigram的算法思想是**从一个巨大的词汇表出发**，再**逐渐删除其中的词汇**，直到size满足预定义。

初始的词汇表可以**采用所有预分词器分出来的词，再加上所有高频的子串**。
每次从词汇表中删除词汇的**原则是使预定义的损失最小**。训练时，计算loss的公式为：
$$
Loss=-\sum_{i=1}^N log(\sum_{x\in S(x_i)}p(x))
$$
假设训练文档中的所有词分别为 $x_1,x_2,...,x_N$，而**每个词tokenize的方法**是一个集合 $S(x_i)$。 当一个词汇表确定时，每个词 tokenize 的方法集合 $S(x_i)$ 就是确定的，而每种方法对应着一个概率 $p(x)$。
如果从词汇表中删除部分词，则某些词的 tokenize 的种类集合就会变少，$log(*)$ 中的求和项就会减少，从而增加整体loss。

Unigram算法每次**会从词汇表中挑出使得loss增长最小的10%~20%的词汇**来删除。一般Unigram算法会与SentencePiece算法连用。

### 8.5 SentencePiece

SentencePiece，顾名思义，它是**把一个句子看作一个整体，再拆成片段**，而没有保留天然的词语的概念。一般地，它**把空格space也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表**。

比如，XLNetTokenizer就**采用了_来代替空格**，解码的时候会再用空格替换回来。

目前，Tokenizers库中，所有使用了SentencePiece的都是与Unigram算法联合使用的，比如ALBERT、XLNet、Marian和T5.
