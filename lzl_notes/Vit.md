# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

## 1 背景

transformer 在自然语言处理被广泛应用，而且效果很好，但是在视觉方面的应用仍然很有局限。

前者都是 transformer 与 卷积结合，或者每个像素点，或者只横向和纵向做自注意力。

本文把 transformer 直接应用于 CV，并且对总体结构不作改动。



## 2 模型



![image-20230406150013213](note_images\image-20230406150013213.png)



==图片输入== 

由于 transformer 的输入是 sequence，所以需要对原图片进行处理。

1. 首先将原始图片 resize 为 $H\times W$，即 图片 $x\in \mathbb{R}^{H\times W\times C}$ ，指数分别是 高 宽 通道数，（实验中 $H=W=224,\ C=3$ ）
2. 将原图片切分为若干个 patches，每个 patch 的尺寸为 $P\times P$，（实验中 $P=16$，因此有 $(224\times 224) / (16\times 16)=196$ 个 patches）
3. 那么图片将被切分为 $N$ 个，$N=HW/P^2$，原图片变为 $x_p=\mathbb{R}^{N\times (P^2\cdot C)}$，$N$ 也是输入 有效的输入 transformer 的序列的长度，（实验中，$N=196$  ）

4. Linear Projection of Flattened Patches，将 patches 摊平，然后进行线性投影，映射为一个 $D$ 维向量，也就是将 patch 拉直，尺寸为 $1\times 1 \times D$，长度为 D，本文称为 **patch embedding**，对应公式如下。(实验中，$D=P\times P \times C=16*16*3=768$)

   

5. 添加一个 $\mathbf{z}_0^0 = \mathbf{x}_{ class}$ ，就像 bert 的 [CLS], 最后的输出为 $\mathbf{z}_L^0$，代表图片的表征 $\mathbf{y}$，可以被应用在微调的图像分类
   $$
   \mathbf{y} =\operatorname{LN}\left(\mathbf{z}_{L}^{0}\right)
   $$

6. position embedding 用来保留 patches 的位置信息，直接使用 1D 位置信息，因为在实验中发现 2D 位置信息没用，（实验中，每个位置是一个 $1\times 768$ 的向量，直接与 patch embedding 相加，维度是 197*768，然后输入 transformer）

$$
\mathbf{z}_{0} =\left[\mathbf{x}_{\text {class }} ; \mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}\right]+\mathbf{E}_{\text {pos }} \ \ \ \ \ \ \ \ \mathbf{E} \in \mathbb{R}^{\left(P^{2} \cdot C\right) \times D}, \mathbf{E}_{\text {pos }} \in \mathbb{R}^{(N+1) \times D}
$$

$\mathbf{z_0}$ 代表第 0 层，即将要输入 encoder 的层。

在 transformer 中：
$$
\begin{aligned}
\mathbf{z}_{\ell}^{\prime} & =\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}_{\ell-1}\right)\right)+\mathbf{z}_{\ell-1}, & & \ell=1 \ldots L \\

\mathbf{z}_{\ell} & =\operatorname{MLP}\left(\operatorname{LN}\left(\mathbf{z}_{\ell}^{\prime}\right)\right)+\mathbf{z}_{\ell}^{\prime}, & & \ell=1 \ldots L
\end{aligned}
$$
$\text{MSA}$ 代表 multi-headed self-attention，$\text{LN}$ 代表 layer norm





## 3 实验

Transformers 缺乏 CNN 固有的一些归纳偏置(inductive biases)，例如平移同变性（translation equivariance）和局部性，因此在数据量不足的情况下不能很好地泛化。但是在大数据集里面效果特别好。





## 4 结论

本文把 transformer 应用于视觉分类任务，效果很好，为以后 transformer 应用于 CV 开启了先河。









