# KVPFormer

## 1 背景 

提取表格类型的文档图片(form-like document image)上的 Key-Value 信息。

不是预训练模型

将 KV 任务转化为问答任务



## 2 问题描述

给定一个 form-like document image D，D 有 $n$ 个实体，$D = [E_1,E_2,...,E_n]$ 

$E_i$ 由一组单子构成 $w_i^1, w_i^2, ... ,w_i^m$，$E_i$ 的 bounding box 为 $[x_i^1,y_i^1,x_i^2,y_i^2]$，左上角和右下角

$E_i\longrightarrow E_j$ 表示 key => value

任务的目标是提取图片中所有的 KVPs（Key-Value pairs）



==本文方法== 

将 key-value pair 提取问题转化为 QA 问题：

基于 Transformer 的 encoder-decoder，首先用 encoder 提取出所有的 key，然后将 keys 作为 questions，喂进 decoder，去预测 key 对应的 value。



### 3 模型

KVPFormer 由三个关键内容：

> 1. 实体表征提取模块：对每个实体，提取一个向量表征
> 2. 基于问题识别的 transformer encoder 模块：在所有的实体中去识别哪个是 key
> 3. 基于由粗到细的答案预测 transformer decoder 模块：对每个 key ，去预测对应的 value

![image-20230419152821156](note_images\image-20230419152821156.png)



### 3.1 实体表征提取

使用 LayoutLM 和 LayoutLM 去提取每个实体的表征，

为了评估方法的上界表现(upper bound performance)，将实体标签的 embedding 拼接（concatenate）到实体表征中。

​        $c_i=e_i\oplus l_i$      $e_i$ 是实体表征， $l_i$ 是可学习的表示实体 label 的embedding，

concatenate 是指两个 embedding 拼起来,例如 $e=[x_1,x_2,...x_n]$，$l=[y_1,y_2,...y_n]$，有很多种拼接的方式，但是拼接后是，$x_i$ 和 $y_i$ 都是不会消失的，拼接的结果的尺寸是 $e$ 和 $l$ 之和。



### 3.2 Question key 的识别

   将所有的实体表征喂进 transformer encoder，为了清楚的建模不同实体间的空间关系，引入新的空间感知自注意力机制到 transformer 的自注意力层，具体来说，就是将 输入到每个 encoder 层的实体表征 作为内容 embedding，然后与其对应的 2D 位置 embedding 进行拼接，以获得每个多头 self-attetion层的 key 和 query embedding，value 是不需要拼接的。

​    另外引入 spatial compatibility attention bias 以建模实体间的空间相容关系。

**注意力分数**为：
$$
\begin{array}{l}
\alpha_{i j}&=\left(\mathbf{c}_{q_{i}} \oplus \mathbf{p}_{q_{i}}\right)^{T}\left(\mathbf{c}_{k_{j}} \oplus \mathbf{p}_{k_{j}}\right)+\mathbf{F} \mathbf{F N}\left(\mathbf{r}_{q_{i}, k_{j}}\right) \\
&=\mathbf{c}_{q_{i}}^{T} \mathbf{c}_{k_{j}}+\mathbf{p}_{q_{i}}^{T} \mathbf{p}_{k_{j}}+\mathbf{F} \mathbf{F N}\left(\mathbf{r}_{q_{i}, k_{j}}\right)
\end{array}
$$
$\mathbf{c}_{q_{i}}\ / \ \mathbf{c}_{k_{j}}$ 和 $\mathbf{p}_{q_{i}}\ / \ \mathbf{p}_{k_{j}}$ 代表 **query $q_i$** 和 **key $k_j$** 的内容 和 2D位置 embedding

$\mathbf{F} \mathbf{F N}\left(\mathbf{r}_{q_{i}, k_{j}}\right)$ 代表 spatial compatibility attention bias，FFN 是一个 两层 的 feedforward 网络，$\mathbf{r}_{q_{i}, k_{j}}$ 是 spatial compatibility feature vector，由拼接的 6-d 向量组成:
$$
\mathbf{r}_{q_{i}, k_{j}}=\left(\Delta\left(B_{i}, B_{j}\right), \Delta\left(B_{i}, U_{i j}\right), \Delta\left(B_{j}, U_{i j}\right)\right)
$$
$B_{i}, B_{j}$ 分别代表 $q_i$ 和 $k_j$ 的 bounding boxes，$U_{ij}$ 代表 $B_{i}, B_{j}$ 的并集，$\Delta\left(B_{i}, B_{j}\right)$ 代表 $B_{i} 和 B_{j}$ 之间的差，例如，

$\Delta\left(B_{i}, B_{j}\right)=(t_{ij}^{x_{ctr}},t_{ij}^{y_{ctr}},t_{ij}^{w},t_{ij}^{h},t_{ji}^{x_{ctr}},t_{ji}^{y_{ctr}})$，每个部分计算方式如下：
$$
t_{i j}^{x_{\mathrm{ctr}}}&=\left(x_{B_{i}}^{\mathrm{ctr}}-x_{B_{j}}^{\mathrm{ctr}}\right) / w_{B_{i}}, \quad t_{i j}^{y_{\mathrm{ctr}}}&=\left(y_{B_{i}}^{\mathrm{ctr}}-y_{B_{j}}^{\mathrm{ctr}}\right) / h_{B_{i}}, \\
t_{i j}^{w}&=\log \left(w_{B_{i}} / w_{B_{j}}\right), \quad t_{i j}^{h}&=\log \left(h_{B_{i}} / h_{B_{j}}\right) \text {, } \\
t_{j i}^{x_{\mathrm{ctr}}}&=\left(x_{B_{j}}^{\mathrm{ctr}}-x_{B_{i}}^{\mathrm{ctr}}\right) / w_{B_{j}}, \quad t_{j i}^{y_{\mathrm{ctr}}}&=\left(y_{B_{j}}^{\mathrm{ctr}}-y_{B_{i}}^{\mathrm{ctr}}\right) / h_{B_{j}},
$$
$\left(x_{B_{i}}^{\mathrm{ctr}}，x_{B_{j}}^{\mathrm{ctr}}\right)$ 代表 bounding 的中心坐标，$w_{B_i}$ 代表 宽，$h_{B_i}$ 代表高。

每一个 feature 从 self-attention 输出是 所有带有正则化注意力分数的 value 的内容 embedding 加权和。
$$
\mathbf{z}_i = \sum_j \frac{exp(\alpha_{ij})}{\sum_k exp(\alpha_{ik})}\mathbf{c}_{v_j}
$$
==识别实体 key 的类型== 

上述通过 transformer encoder 得到一个增强的 实体表征，然后加一个线性层去识别 key 实体，这是一个二分类任务。



### 3.3 由粗到细的答案预测

可以并行地将所有问题表征喂进 transformer decoder，在每一个 decoder 层，

> **DETR-style deccoder** 
>
> DETR（DEtection TRansformer）是一种基于Transformer的端到端目标检测模型，由Facebook AI Research团队提出。它使用Transformer将目标检测任务转换为一种集合预测问题（set prediction），即将输入的图像和目标集合编码为两个集合，然后通过匹配这两个集合来预测目标的类别、位置和数量。
>
> 由大概到精确的过程。



形式上

$\mathbf{H}=[\mathbf{h}_1,\mathbf{h}_2,...,\mathbf{h}_N]$ 代表 encoder 输出的实体表征，N 是实体的数量

$\mathbf{Q}=[\mathbf{q}_1,\mathbf{q}_2,...,\mathbf{q}_M]$ 代表 decoder 输出的问题表征，M 是识别出问题也就是 key 的数量

在 coarse（粗糙） 阶段，使用二分类计算 $s_{ij}^{coarse}$ 去计算 $\mathbf{h}_j$ 是答案(value)的可能性
$$
s_{ij}^{coarse}=\text{Sigmoid}(\text{MLP}_{coarse}(\mathbf{q}_i+\mathbf{h}_j+\mathbf{FFN}(\mathbf{r}_{ij})))
$$
$\mathbf{r}_{ij}$ 是 $\mathbf{q}_i$ 和 $\mathbf{h}_j$ 之间的 18-d spatial compatibility feature



对于每一个 question $\mathbf{q}_i$，降序排序 score为 $[s_{ij},j=1,2,...,N]$，然后选择 K=5 个分数最高的候选答案，$[h_{i_k},k=1,2,...,K]$ 

在 精确(fine) 阶段，使用多类分类器计算 $\mathbf{q}_i$ 和 $\mathbf{h}_j$ 之间的 $s_{ij}^{fine}$ 
$$
t_{ii_k}=\text{MLP}_\text{fine}(\mathbf{q}_i+\mathbf{h}_{i_k}+\mathbf{FFN}(\mathbf{r}_{ii_k}))\\
s_{ij}^{fine}=\frac{exp(t_{ii_k})}{\sum_kexp(t_{ii_k})}
$$

### 3.4 Loss 函数

损失分为两个部分，第一个是从实体中选择 key，第二个是找到 key 对应的 value。



**Question Identification Loss：** 交叉熵损失
$$
\mathcal{L}_Q=\frac{1}{N}\sum_i CE({p}_i,{p}_i^*)
$$
​    $p_i$ 是预测 $i^{th}$ 实体的结果，$p_i^*$ 是label。



**Coarse-to-Fine Answer Prediction Loss：** 二进制交叉损失熵
$$
\mathcal{L}_A^{coarse}=\frac{1}{MN}\sum_{ij} BCE({s}_i^{coarse},{s}_i^{coarse^*})
$$
${s}_i^{coarse^*}\in \{0,1\}$ 是 ground-truth，实体是答案的时候，为 1，否则为 0。

总的损失是：
$$
\mathcal{L}=\mathcal{L}_Q+\mathcal{L}_A^{coarse}+\mathcal{L}_A^{fine}
$$


## 4 实验

FUNSD 和 XFUND



一个 key 对应多个 value 的情况：识别出所有的 value，去预测 key，反过来预测。

[Entity Relation Extraction as Dependency Parsing in Visually Rich Documents](Entity Relation Extraction as Dependency Parsing in Visually Rich Documents.md)

Note that on FUNSD and XFUND datasets, since one key entity may have multiple corresponding value entities while each value entity only has at most one key entity, we follow (Zhang et al. 2021) to first identify all the value entities as questions and then predict their corresponding key entities as answers to obtain key-value pairs.













