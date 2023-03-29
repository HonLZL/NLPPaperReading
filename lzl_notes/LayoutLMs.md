## 1 文档智能

### 1.1 文档智能介绍

在过去的 30 年中，文档智能的发展大致经历了三个阶段:

1. 基于启发式规则

通过人工观察文档的布局信息，总结归纳一些处理规则，对固定布局信息的文档进行处理

2. 基于统计机器学习

3. 基于深度学习，多模态预训练技术

   



![image-20230323161959454](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230323161959454.png)



要求能充分利用文档的视觉信息和文本信息

视觉信息：

字体 字号 字形等，文本位置，全图结构



文本信息：

文字的语义，文字位置的语义



==四大类任务== 

1. 文档版面分析：

   指对文档版面内的图像、文本、表格信息和位置关系所进行的自动分析、识别和理解的过程。

   

2. **文档信息抽取**：

   指从文档中大量非结构化内容中抽取实体及其关系的技术。与传统的纯文本信息抽取不同，文档的构建使得文字由一维的顺序排列变为二维的空间排列，因此文本信息、视觉信息和位置信息在文档信息抽取中都是极为重要的影响因素。

   

3. 文档视觉问答：

   指给定文档图像数据，利用 OCR 技术或其他文字提取技术自动识别影像资料后，通过判断所识别文字的内在逻辑，回答关于图片的自然语言问题。

   

4. 文档图像分类：

   指针对文档图像进行分析识别从而归类的过程。



### 1.2 数据集介绍

#### 1.2.1 FUNSD

FUNSD 是一个文档理解数据集，其包含 199 张表单，每张表单中包含表单实体的键值对。

| <img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\86220490.png" alt="86220490" style="zoom: 50%;" /> | 原始格式 ：{<br/>    "form": [{},{},···{}]<br/>}<br/>form里每个 {} 都是一个 box<br/>里面有文本的bounding box<br/>具体内容 text<br/>label，是 question或answer <br/>以及id，等等<br/>最重要的是 linking，[x, y]<br/>指的是 key 的 id 和 value 的 id |
| ------------------------------------------------------------ | ------------------------------------------------------------ |







## 2 LayoutLM

### 2.1 背景

富文本文档主要包含三种模态信息：文本、布局以及视觉信息，并且这三种模态在富文本文档中有天然的对齐特性。因此，如何对文档进行建模并且通过训练达到跨模态对齐是一个重要的问题。

这些模态对齐的富文本格式所展现的视觉特征可以通过视觉模型抽取，结合到预训练阶段，从而有效地帮助下游任务。

预训练技术被广泛的应用于 NLP 任务，针对于文本。而在文档中，布局和样式等信息同样是很重要的。

![image-20230323201414873](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230323201414873.png)



### 2.2 模型



除了文本信息外，LayoutLM 还关注两个方面的特征：

> ***Document Layout* ** 
>
> ​	文本间的相对位置能给出大量的语义信息，例如 key 和 value 对，value 通常在 key 的右边和下边，因此可以将相对位置信息作为 2-D 位置特征，可以使用 self-attention 编码。
>
> ***Visual Information* ** 
>
> ​	文档内包含着许多视觉信息，能显示文档片段的重要性和优先级，例如字体加粗。对于整个文档而言，视觉包含着文档结构，可以作为文档分类；对于文本而言，有字体加粗，下划线，斜体等，对文档理解都是有意义的。



对于模型结构，使用 BERT 作为 backbone，另外新增两个输入 embedding，2-D position embedding 和 image embedding。

需要注意的是，Image Embedding 和其他是分隔开的，Image Embedding 在 Layout 预训练之后加的。

![image-20230324094503830](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230324094503830.png)



> **2-D Position Embedding** 
>
> ​	将文档左上角作为原点，那么一个 bounding box 可以被定义为 $(x_0,y_0,x_1,y_1)$ 。上图显示，添加了 4 个位置 embedding，最终的 2-D　Position　Embedding 为四个子层的 Embedding 之和。可以通过 OCR 或者 PDF 解析获得 bounding box。因为图像像素差距较大，所以需要将坐标 resize 1000内，并取整。
>
> Image Embedding
>
> ​	使用每个文本的 Bounding Box 分割图片为多个部分，分割的部分与文本一一对应，然后将每个部分通过 Faster R-CNN 生成图像区域特征，将其作为 token image embedding。
>
> ​	[CLS] 符号用于表示整个输入文本的语义，同样使用整张文档图像，利用 Faster R-CNN 作为 ROI 生成的 Image Embedding，从而保持模态对齐。有利于需要 [CLS] 的下游任务。



### 2.3 模型预训练与微调

1. **Masked Visual-Language Model (MVLM)** 掩码式语言模型

​	大量实验已经证明 MLM 能够在预训练阶段有效地进行自监督学习。在此模型 MVLM 基础上进行了修改：在遮盖 (MASK) 当前词之后，保留对应的 2-D　Position　Embedding 暗示，让模型预测对应的词。在这种方法下，模型根据已有的上下文和对应的视觉暗示预测被遮罩的词，从而让模型更好地学习文本位置和文本语义的模态`对齐关系`。

2. **Multi-label Document Classification (MDC)** 多标签文档分类

​	MLM 能够有效地表示词级别的信息，但是对于文档级的表示，需要文档级的预训练任务来引入更高层的语义信息。在预训练阶段使用 IIT-CDIP 数据集为每个文档提供了多标签的文档类型标注，同时引入 MDC 多标签文档分类任务。该任务使得模型可以利用这些监督信号去聚合相应的文档类别，并捕捉文档类型信息，从而获得更有效的高层语义表示。



**微调** 

微调了三个任务：表达理解，发票理解，文档图片分类。

对于表单和发票理解，模型对每个 token 预测出 {B,I,E,S,O} 标签，然后使用到数据库中检测每个实体的类型。

对于文档图片分类任务，模型预测从过 [CLS] 的表征去预测类标签。



### 2.4 局限 

1. 没有将视觉信息与文本和layout信息进行深度融合，而是分成两步，简单的家在了一起
2. 没有将视觉信息与文本和layout信息进行对齐
3. 预训练任务不够丰富
4. 根据 NLP 的预训练模型经验，采用相对位置编码可以提升模型效果
5. 文档分类数据集是有监督的，所以较少



## 3 LayoutLMv2



相比于 LayoutLM，v2 不仅使用 MVLM 预训练任务，还新增了新的文字图片与文字图片对齐任务，这将能更好地在与训练阶段获得跨模态交互信息。

- 将视觉信息加入到了预训练阶段，而不是LayoutLM中的微调阶段
- 删除了MDC，添加了text-image alignment和text-image matching两个预训练任务
- 将spatial-aware的自注意力机制整合到了transformer中

**Visually-rich Document Understanding (VrDU)** 富文本文档



### 3.1 模型架构

==Visual/Text Token Embedding== 

**Text Embedding** 

采用 `UniLM-v2` 模型的 embedding 层初始化，初始化方式如下

使用 WordPiece 将 OCR 得到的文本序列分词，每个 token 都分配一个片段 $s_i\in \{[A],[B]\}$，将 [CLS] 放在文本 序列前面，[SEP] 放在后面，如果需要 padding，就 [PAD] 放在末尾以保持对齐。

最终的 text Embedding 是三个 Embeddings 之和。

1. token embedding 代表 token 自己
2. 1D positional embedding 代表 toekn 的下标
3. segment embedding 被用来区分不同的文本片段

$$
t_i=\text{TokEmb}(w_i)+\text{PosEmb1D}(i)+\text{SegEmb}(s_i)
$$



**Visual Embedding**  

使用 `ResNeXt-FPN` 进行视觉编码器，文档图片都被 resize 到 224×224

7×7 个视觉 tokens， 将图片拆分成 7*7 个区域，分别编码。（模型结构图中以 2×2 为例）

全连接层将视觉 token 的维度与 token embedding 适配

视觉编码在 预训练过程 中也进行参数更新。

输出的 feature map 是 宽 $W$，高 $H$，然后它被摊平到一个长度为 $W\times H$ 的 visual embedding 序列，叫做 \$\text{VisTokEmb}(I)$，然后对其添加一个线性投影层 $\text{Proj()}$ 去与 text embedding 的维度统一。

因为 CNN 视觉 backbone 不能获得位置信息，所以再添加一个 1D positional embedding。这个 embedding 是与 text embedding 共享的。

对于 segment embedding，我们将所有的视觉 token 附加到视觉片段 **[C]** 上。
$$
\mathbf{v}_i=&\text{Proj}(\text{VisTokEmb}(I)_i)\\
&+\text{PosEmb1D}(i)+\text{SegEmb([C])}
$$




将上述的 **Visual Embedding** 和 **Text Embedding** 左右拼接得到 **Visual/Text Token Embedding** 

==1D Position Embedding== 

**Visual token 部分**: 位置编码为从 0 到 7*7-1 （模型结构图中是 0 到 2×2-1）

**Text 部分** : 以 [CLS] 为起点，从 0 开始



==2D Position Embedding== **layout 部分** 

LayoutLM 采用文本框的左上角和右下角，没用更明显的表现出文本框的大小。所以 v2 采用以下方法：

对每个 token 的 bounding box：$(x_{min}, y_{min}), (x_{max}, y_{max})$ 
$$
I_i=Concat(&PosEmd2D_x(x_{min},x_{max}, width),\\
&PosEmd2D_y(y_{min},y_{max}, height))
$$
对于特殊字符，假设其 box 退化为左上角顶点

对于视觉 token，根据 CNN 的性质，视觉 token 对应于原图的 7*7 的网格，采用网络的坐标形成位置编码

一个空的 bounding box $\text{box}_\text{PAD}=(0,0,0,0,0,0)$ 被添加特殊token [CLS], [SEP], [PAD]。



==带有空间信息的多模态编码器== 

visual embedding $\{\mathbf{v}_0, ···,\mathbf{v}_{WH-1}\}$ 

text embedding $\{\mathbf{t}_0, ···, \mathbf{t}_{L-1}\}$ 

编码器将以上两个 embedding 拼接为统一的序列，通过添加 layout embedding 来融合空间信息。
$$
\mathbf{x}_i^{(0)}=X_i+I_i, where\\
X=\{\mathbf{v}_0,···,\mathbf{v}_{WH-1},\mathbf{t}_0,···,\mathbf{t}_{L-1}\}\
$$
最初的注意力机制的注意力分数为
$$
\alpha_{ij}=\frac{1}{\sqrt{d_{head}}}(\mathbf{x}_i \mathbf{W}^Q)(\mathbf{x}_j \mathbf{W}^K)^{\top}
$$
由于位置太多了，不够突出这个特征，所以将语义相对位置和空间相对位置作为偏置项，以防止添加过多参数。

让 $\mathbf{b}^{1D}$ ， $\mathbf{b}^{2D_x}$ 和 $\mathbf{b}^{2D_y}$ 表示为 1D 和 2D 相对位置偏置的可学习参数。因此注意力分数变为：
$$
\alpha_{ij}^{'}=\alpha_{ij}+ \mathbf{b}^{1D}_{j-i} +\mathbf{b}^{2D_x}_{x_j-x_i}+\mathbf{b}^{2D_y}_{y_j-y_i}
$$
最终的输出向量为 
$$
\mathbf{h}_{i}=\sum_{j} \frac{\exp \left(\alpha_{i j}^{\prime}\right)}{\sum_{k} \exp \left(\alpha_{i k}^{\prime}\right)} \mathbf{x}_{j} \mathbf{W}^{V}
$$


![image-20230325184420091](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230325184420091.png)

![image-20230325184454835](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230325184454835.png)



### 3.2 预训练任务

1. **Masked Visual-Language Modeling （MVLM）** 和 LayoutLM 一样，随机 mask 一些 text tokens，但是这个 token 的 2D position仍然保留，然后询问模型去复原 mask 的tokens，鼓励模型采用 layout 信息辅助 token 补全。但是在 v2 中，layout 信息也被添加进去了，所以也需要 mask 掩码部分在原图的 bounding box。

2. **Text-Image Alignment （TIA）** 帮助模型学习 空间位置和 bounding boxes 的对齐

   随机选择一整行文字，图片上的一些token lines会被覆盖（cover）掉，然后使用对应的text token预测图片中的token line是否被覆盖。

   在计算TIA Loss的时候，被遮蔽（mask）的text token不会参与计算。

3. **Text-Image Matching** 

   判断文本和图片是否匹配，即在该文档里（还是在其它文档里），预训练时会构造负样本（替换文档或丢弃文档），正负样本使用同样的覆盖和遮蔽操作。最后通过[CLS]预测是否匹配，不匹配的话text token全部为“已覆盖（Covered）”

添加 [Covered] 或者 [Not Covered] 是为了避免预训练任务的混淆。























## 4 LayoutLMv3

































