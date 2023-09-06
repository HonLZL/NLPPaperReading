# DiT: Self-supervised Pre-training for Document Image Transformer

## 1 背景

有大量未标注的文档图片可以使用。文档图像任务的与训练是必要的。



下游任务：document image classification, document layout analysis, table detection and text detection for OCR





## 2 模型架构



![image-20230802170731265](note_images\image-20230802170731265.png)



类似 ViT，使用 transformer 架构作为 DiT 的backbone，首先将文档图片划分为不重叠的 patches，获得对应的 patch embedding，再加上 1d position embedding，这些图片 patches 被送进有多头注意力机制的 transformer 块，最终输出 image patch 的表征。

==预训练任务== 

类似 BEiT，使用 Masked Image Modeling（MIM）作为预训练任务。

使用 DALL-E 里面的 dVAE 作为 image tokenizer，然而，原本的 DALL-E tokenizer 是用自然图片预训练的，这与文本图片有较大差异，因此本文为了得到更好的从文档图片分离出 visual tokens，本文在 IIT-CDIP 继续预训练 dVAE。

为了有效地预训练 DiT 模型，我们在给定一系列图像 patches 的情况下，使用特殊标记[MASK]随机屏蔽一个输入子集。DiT encoder 用加入位置 embedding 的线性投影生成 patch sequence 的 embedding，然后经过一堆 transformer 块。

模型需要预测 mask 位置的 visual token 的下标。



## 3 微调

RVL-CDIP：文档图片分类

PubLayNet：布局分析

ICDAR 2019 cTDaR：表格检测

FUNSD：文本检测

以上任务可以被归为图片分类和目标检测。

**图片分类**：使用平均池化来聚合图像块的表示。接下来，将全局表示传递到一个简单的线性分类器中。

**目标检测**：如下图，先在四个不同的 transformer blocks 使用分辨率修正模块，使单尺度 ViT 适应多尺度 FPN。$d$ 是总的 blocks 的数量:

$\frac{d}{3}$​ 块使用 2stride 的 2*2的转置卷积核进行 4 倍上采样。

对上步输出的 $\frac{d}{2}$ 块使用 1stride 的 2*2的转置卷积核进行 4 倍上采样。

对上步输出的 $\frac{2d}{3}$ 块使用直接输出。

最终，对第三步的输出的所有块，进行 2 倍下采样，并进行 2×2 的最大池化处理。

对 FPN 的输出，再经过一个 Detection 框架，得到检测结果。

![image-20230802175225865](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20230802175225865.png)





> **FPN 是什么？** 
>
> FPN（Feature Pyramid Network 特征金字塔网络）是一种用于解决目标检测和语义分割任务中的深度学习架构。它通过在网络中创建金字塔形的特征图集合，从不同分辨率的特征图中提取多尺度的语义信息。
>
> 在传统的深度卷积神经网络中，特征图的分辨率逐渐降低，导致较低分辨率的特征图丢失了大量的细节信息，这对于目标检测和语义分割任务非常不利。FPN的目标就是解决这个问题。
>
> FPN的核心思想是通过自上而下和自下而上的信息传播来构建特征金字塔。首先，从底层到顶层，通过上采样将低分辨率的特征图上升到高分辨率，称为自下而上路径。然后，从高层传递到低层，通过降采样将高分辨率的特征图下降到低分辨率，称为自上而下路径。通过这种方式，FPN能够融合来自不同分辨率的特征图，并提取多尺度的语义信息。
>
> 此外，FPN还引入了一个额外的侧重分支（top-down pathway），用于进一步提高信息的传递效果。该分支通过将高分辨率的特征图与相应低分辨率的特征图进行融合，得到更加丰富的特征表示。
>
> FPN被广泛应用于目标检测和语义分割任务中。在目标检测中，FPN能够提供多尺度的特征来检测不同大小的目标。在语义分割中，FPN能够提供详细的物体边界信息并准确地分割出每个物体。
>
> 总的来说，FPN通过特征金字塔的构建和特征融合，能够有效地解决目标在不同尺度上的检测和分割问题，提高了模型的性能和泛化能力。
>
> ![image-20230802213655073](note_images\image-20230802213655073.png)

## 

























































