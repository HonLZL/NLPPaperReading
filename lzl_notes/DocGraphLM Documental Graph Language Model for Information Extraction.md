# DocGraphLM: Documental Graph Language Model for Information Extraction

## 1 背景

使用 transformer 的模型在表示空间距离较远的语义方面存在挑战，例如表的元素距离表头较远，或者内容跨越换行符。针对这些情况，有研究提出使用图神经网络对文档中文标记或片段之间的关系和结构进行建模，虽然这些模型本身仍然不如布局语言模型，但它们展示了结合额外的结构化信息改进文档表示的潜力。

文档智能主要有两种模型，预训练和图神经网络。

受以上启发，本文采用将预训练语言模型和图语义相结合的新框架

本文的贡献：

1 提出了新颖的架构将图神经网络与预训练的语言模型集成在一起，以增强文档表示

2 将链接预测方法引入到文档图重建中，并提出了联合损失函数，该损失函数强调对邻近邻居节点的恢复



## 2 模型

![image-20231030163810699](note_images\image-20231030163810699.png)



### 2.1 将文档表示为图

图由点和边组成，点代表文本信息，边代表文本之间的关系。

为了生成变，使用新的启发式的 Direction Line-of-Sight（D-LoS），代替 K近邻，K近邻可能会导致不相关的行列被作为邻居，忽略了距离较远的 KV 对。

D-LoS 将节点的周围360度分为 8 个45度扇区，然后确定 8 个扇区最近的节点，8 个扇区定义了 8 个方向。

**节点表示：**  

节点由两个特征：文本特征和节点size，语义信息从语言大模型获得，节点size由坐标获得：
$$
M=\text{emb}([width, height]),\ width=x_2-x_1\ and\ height=y_2-y_1 
$$
$(x_1,y_1)$ 和 $(x_2, y_2)$ 是文本的左上点和右下点。凭直觉来看，节点大小是一个重要的指标，有助于区分字大小二和潜在的段落的语义角色，如title、caption、body。

节点的表示为：
$$
E_u = \text{emb}(T_u),\ \ u={1,2,...,N}
$$
**边的表示：** 

使用极坐标特征，包括关系距离和方向，先计算两个 box 的最短的欧几里得距离 $d$，为了减少语义不相关的远距离节点的影响，使用距离平滑技术，$e_{dis}=log(d+1)$，由 D-LoS 得到的一对节点的方向：$e_{dir}\in \{0,1,...,7\}$，定义一份链接 $e_p=[d_{dis},e_{dir}]$，区重建文档图。



## 2.2 通过链接预测重构图

​	通过预测链接的预测两个关键属性 $e_p$ 来重建属性，并将这一过程定义为多任务学习问题。

GNN 的输入是编码后的节点表示，该表示通过 GNN 的消息传递机智传递：
$$
h_u^{G,l+1}:=\text{aggregate}(h_v^{G,l},\forall v\in \mathcal{N} (u))
$$
$l$ 是邻居层，$\mathcal{N} (u)$ 代表 $n$ 的邻居节点几何，$\text{aggregate}(·)$ 是 GNN 聚合函数，更新节点表示。

​	在两个任务上联合训练 GNN，预测节点之间的距离和方向。

​	对于距离预测，定义分类回归头 $\hat{y}_{u,v}^e$，通过两个节点向量的点积来生成标量值，并使用线性激活：
$$
\hat{y}_{u,v}^e = Linear((h_u^G)^{\top}\times h_v^G)
$$
​	对于方向预测，定义分类头 $\hat{y}_{u,v}^d$，根据两个节点的 element-wise product（对应元素的乘积，哈达玛积），为每条边分配八个方向：
$$
\hat{y}_{u,v}^d = \sigma ((h_u^G \odot  h_v^G )\times W)
$$
​	都使用 MSE loss，联合 loss 为: 
$$
\begin{array}{l}
\text { loss }=\sum_{(u, v) \in \text { batch }}\left[\left(\lambda \cdot \operatorname{loss}^{\mathrm{MSE}}\left(\hat{y}_{u, v}^{e}, y_{u, v}^{e}\right)\right.\right. \\
\left.+(1-\lambda) \cdot \operatorname{loss}^{\mathrm{CE}}\left(\hat{y}_{u, v}^{d}, y_{u, v}^{d}\right)\right] \cdot\left(1-r_{u, v}\right)
\end{array}
$$


$\lambda$ 是超参数，用来平衡两个 loss 的权重，$r_{u,v}$ 是距离 $e_{dis}$ 的归一化，在[0,1]，因此 $r_{u,v}$ 的值降低了远处片段的权重，使模型更关注附近片段。



### 2.3 图神经网络和预训练模型输出的联合表示



$h_u^C=f(h_u^L,h_u^G)$ 是联合表示，其中 $h_u^L$ 是 语言模型 表示，$h_u^G$  是 GNN 表示，函数 $f$  表示 拼接，平均或求和。本文采用 token 级别的拼接。

引入的节点表示可以作为其他模型的输入，以促进下游任务。例如实体抽取的 $\text{IE\_head}(h_u^C)$，视觉问答任务 $\text{QA\_Head}(h_u^V)$ 。



## 三 实验



![image-20231030204156144](note_images\image-20231030204156144.png)































































