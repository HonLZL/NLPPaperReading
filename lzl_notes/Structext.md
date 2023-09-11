# StrucTexT

## 一 StrucTexT: Structured Text Understanding with Multi-Modal Transformers

### 1 背景

传统的实体标注方法当作一个序列标注问题，文本段被序列化一个有序的线性序列，使用 IOB（Inside, Outside, Beginning） 标注方法，然而能力有限，因为是在 Token 级别的。

tan woon yann 

B-HEADER   I-HEADER    I-HEADER 

如图所示，b 和 c 图，文档包含的信息通常是以 Segment 组成。 Segment 呈现的几何和语义信息更丰富。

还有一种将实体标注专注于 Segment 表示，无法处理由字符组成的实体。

因此本文将 Token 和 Segment 都考虑在内。

没考虑文本段包含多个类别；



![image-20230909144103244](note_images\image-20230909144103244.png)

指出 LayoutLMv2 对图像中的结构化信息利用不足。



### 2 方法 

![image-20230909155848863](note_images\image-20230909155848863.png)



#### 2.1 embedding

==多模态 embedding== 

bounding box $b_i=(x_0, y_0, x_1, y_1)$ 高为 $h$, 宽为 $w$  

文本  $t_i=\{c_i^i,c_2^i,...,c_{l_i}^i\}$ 

**Layout Embedding：**
$$
L=Emb_l(x_0,y_0,x_1,y_1,w,h)
$$


**Language Token Embedding:** 

使用 WordPiece 去 tokenize 句子



**Visual Segment Embedding：**



**Segment ID Embedding：**



**Other Embeddings：**

position embedding、segment embedding


$$
Input=Concat(T,V)+S^{id}+P^{id}+M^{id}
$$


==预训练任务== 

**MVLM**(Masked Visual Language Modeling)

仿照 ViLBERT，从语言序列选取 15% 的 token，用 [MASK] 遮盖其中的 80%，用随机 token 替换其中的 10%，并保持 10% 的token 不变。然后需要模型预测相应的 token。



**SLP**(Segment Length Prediction)

从每个视觉特征，预测出 文本segment 的长度

作者认为，这种信息流通可以加速文本、视觉和布局信息之间的深度跨模态融合。



**PBD**(Paired Box Direction)

配对文本方向，利用全局的布局信息。

几何拓扑结构.



![image-20230909215438378](note_images\image-20230909215438378.png)

#### 2.2 各模块

**Cross-granularity Labeling Module：**  

同时支持 token 级别的实体标注和 segment 级别的实体标注任务。

文本端有着相同 ID 的 token 通过算数平均聚合为 segment 级别的文本特征。
$$
\hat T_i=mean(\hat t_i )=(\hat c_1+\hat c_2+...+\hat c_{l_i})/l_i
$$




### 3 微调 

我们在三个信息抽取任务上对StrucText进行了微调：片段级别的实体标注和实体链接以及令牌级别的实体标注。

对于基于段的实体标注任务，我们通过算术平均聚合文本句子的token特征，并将视觉特征和文本特征相乘得到段级特征。最后，在softmax层后面紧跟特征，进行片段级类别预测。采用实体层面的F1值作为评价指标。

实体链接任务将两个片段特征作为输入，得到成对关系矩阵。然后我们将关系矩阵中的非对角元素通过一个sigmoid层来预测每个关系的二分类。







## 二 StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training

### 1 背景

当前主流多模态文档理解预训练模型需要同时输入文档图像和OCR结果，导致欠缺端到端的表达能力且推理效率偏低等问题



![image-20230909222729354](note_images\image-20230909222729354.png)



当前主流的文档理解预训练方法大致可分为两类：

a）掩码语言建模（Masked Language Modeling），对输入的掩码文本Token进行语言建模，运行时文本的获取依赖于OCR引擎，整个系统的性能提升需要对OCR引擎和文档理解模型两个部件进行同步优化

b）掩码图像建模（Masked Image Modeling），对输入的掩码图像块区进行像素重建，此类方法倾向应用于图像分类和版式分析等任务上，对文档强语义理解能力欠佳。针对上述两种预训练方案呈现的瓶颈

视觉主导，版面分类，表格检测Z

c）本文提出的 StrucTexTv2：统一图像重建与语言建模方式，在大规模文档图像上学习视觉和语言联合特征表达

### 2 方法 

下图描绘了StrucTexTv2的整体框架，主要包含编码器网络和预训练任务分支两部分。编码器网络，主要通过FPN结构串联CNN组件和Transformer组件构成；预训练分支则包含了掩码语言建模（MLM）和掩码图像建模（MIM）双预训练任务头。

![image-20230910095618261](note_images\image-20230910095618261.png)



**编码器网络 ** 

StrucTexTv2采用CNN和Transformer的串联编码器来提取文档图像的视觉和语义特征。文档图像首先经过ResNet网络以获取1/4到1/32的四个不同尺度的特征图。随后采用一个标准的Transformer网络接收最小尺度的特征图并加上1D位置编码向量，提取出包含全局上下文的语义特征。该特征被重新转化为2D形态后，与CNN的其余三个尺度特征图通过FPN[6]融合成4倍下采样的特征图，作为整图的多模态特征表示。



**掩码语言建模：**

借鉴于 BERT 构建的掩码语言模型思路，语言建模分支使用一个2层的MLP将词区域的ROI特征映射到预定义的词表类别上，使用Cross  Entropy  Loss监督。同时为了避免使用词表对文本序列进行标记化时单个词组被拆分成多个子词导致的一对多匹配问题，论文使用分词后每个单词的首个子词作为分类标签。此设计带来的优势是：StrucTexTv2的语言建模无需文本作为输入。



**掩码图像建模**：

考虑到基于图像Patch的掩码重建在文档预训练中展现出一定的潜力，但Patch粒度的特征表示难以恢复文本细节。因此，论文将词粒度掩码同时用作图像重建，即预测被掩码区域的原始像素值。词区域的ROI特征首先通过一个全局池化操作被压缩成特征向量。其次，为了提升图像重建的视觉效果，论文将通过语言建模后的概率特征与池化特征进行拼接，为图像建模引入“Content”信息，使得图像预训练专注于复原文本区域的“Style”部分。图像建模分支由3个全卷积  Block构成。每个Block包含一个Kernel=2×2，Stride=4的反卷积层，一个Kernel=1×1，以及两个Kernel=3×1卷积层。最后，每个单词的池化向量被映射成一个大小为64×64×3的图像，并逐像素与原本的图像区域做MSE Loss。



**实验结果** 

![c8ef00951efb36bb6175d7f26cdabbde.png](note_images\c8ef00951efb36bb6175d7f26cdabbde.png)



### 三 总结

论文出的StructTexTv2模型用于端到端学习文档图像的视觉和语言联合特征表达，图像单模态输入条件下即可实现高效的文档理解。论文提出的预训练方法基于词粒度的图像掩码，能同时预测相应的视觉和文本内容，此外，所提出的编码器网络能够更有效地挖掘大规模文档图像信息。实验表明，StructTexTv2在模型大小和推理效率方面对比之前的方法都有显著提高。



https://github.com/PaddlePaddle/VIMER/tree/main/StrucTexT/v2/



