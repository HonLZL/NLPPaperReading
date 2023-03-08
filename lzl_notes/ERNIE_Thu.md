# ERNIE: Enhanced Language Representation with Informative Entities

## 1 背景

现有的预训练模型很少考虑到融入知识图谱，作者认为，知识图谱中的信息实体可以借助外部知识增强语言表征。

知识图谱相当于一个通用的知识库，能够为语言理解添加额外的信息。

知识图谱就是描述实体以及实体之间的关系，通常用三元组组织信息。节点-边-节点可以看作一条记录，第一个节点看作主语，边看作谓语，第二个节点看作宾语，主谓宾构成一条记录。比如曹操的儿子是曹丕，曹操是主语，儿子是谓语，曹丕是宾语。

![image-20230305152129597](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230305152129597.png)



例如：Bob Dylan wrote Blowin’ in the Wind in 1962, and wrote Chronicles: Volume One in 2004.

对于这句话，通过已有的知识，我们可以提取出许多信息。没有外部的知识，我们就不知道 Blowin' in the wind 是首歌，Chronicles: Volume One是本书，从而我们很难知道鲍勃迪伦是一位作曲家和作家。所以如果有了丰富的知识，我们能更好的理解语言。



但是要想将外部知识引入到语言表征模型中，有两个挑战

- Structured Knowledge Encoding  结构化信息编码，对于给定的文本，能够有效地提取和编码其在知识图谱中的相关信息事实用于语言表示模型
- Heterogeneous Information Fusion 异构信息融合，语言表示的预训练和知识表示的预训练截然不同，是两个独立的向量空间，因此需要设计一个能够融合词汇、语法、知识的的预训练。

作者提出了 ==ERNIE== （**E**nhanced Language **R**epresentatio**N** with **I**nformative **E**ntities) 来对语料和知识图谱进行预训练。



## 2 ERNIE 的实现

### 2.1 文章参数定义

Token 序列为 $\{w_1,w_2,···,w_n\}$，$n$ 是序列的长度。

实体序列（entity sequence）为  $\{e_1,e_2,···,e_m\}$，$m$ 是序列的长度。$m$ 在大多数情况下不等于 $n$。

$\mathcal{V}$ 表示包含所有 Token 的表。

$\mathcal{E} $ 表示 KG 所有实体的表。

如果一个 token $w \in \mathcal{V}$ 对应于 KG 中一个实体 $e \in \mathcal{E} $ ，可以定义一个对应关系 $f(w)=e$ 。



### 2.2 模型结构

<img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230305183745806.png" alt="image-20230305183745806" style="zoom:67%;" />

ERNIE 由两个堆叠模块组成

1. 文本编码器 T-Encoder，负责从 token input 获得机会和语法信息。T-Encoder 是双向 Transformer 的 encoder，和 bert 一样。

   输入 token 序列 $\{w_1,w_2,···,w_n\}$，经过token embedding, segment embedding, positional embedding的融合得到:
   $$
   \{\boldsymbol{w_1}  ,\boldsymbol{w_2},···,\boldsymbol{w_n} \}=T-Encoder(\{w_1,w_2,···,w_n\})
   $$
   <img src="note_images\image-20221209211656417.png" alt="image-20230305193308231" style="zoom:67%;" />

   

2. 知识编码器 K-Encoder，负责额外的知识信息融合到语言表征(language representation)中，从而将 tokens 和 entities 异构信息融合到到一个统一的特征空间。
   $$
   \{\boldsymbol{w_1^o},\boldsymbol{w_2^o},···,\boldsymbol{w_n^o}\},\ \{\boldsymbol{e_1^o},\boldsymbol{e_2^o},···,\boldsymbol{e_m^o}\}=K-Encoder(\{\boldsymbol{w_1}  ,\boldsymbol{w_2},···,\boldsymbol{w_n}\} ,\ \{\boldsymbol{e_1},\boldsymbol{e_2},···,\boldsymbol{e_m}\})
   $$
   得到的 $\{\boldsymbol{w_1^o},\boldsymbol{w_2^o},···,\boldsymbol{w_n^o}\},\ \{\boldsymbol{e_1^o},\boldsymbol{e_2^o},···,\boldsymbol{e_m^o}\}$，将会被用于具体的任务特征中。



### 2.3 Knowledgeable Encoder

entity embeddings 是通过 TransE 模型预训练得到的，输入各个实体 $\{e_1,e_2,···,e_n\}$，通过预训练 TransE 将实体转化为embedding向量：$ \{\boldsymbol{e_1},\boldsymbol{e_2},···,\boldsymbol{e_m}\}$ 。

目标： f(“Bob Dylan”) + f(“is_a”) = f(“Songwriter”)
其中：h = Bob Dylan, l = is_a, t = Songwriter。同时也要引入负样本，非l两边的节点加入作为h’, t’。（h’, l, t’）构成负样本。

![image-20230305201113665](note_images\image-20230305201113665.png)





<img src="note_images\image-20230305193308231.png" alt="image-20230305193308231" style="zoom:67%;" />

知识解码器中堆叠了 $M$ 个 Aggregators，在第 $i$ 个 Aggregator 中，输入是token embedding $\{\boldsymbol{w_1}  ,\boldsymbol{w_2},···,\boldsymbol{w_n}\} $ 和entity embedding $\{\boldsymbol{e_1},\boldsymbol{e_2},···,\boldsymbol{e_m}\}$。然后分别经过 多头注意力 模块，得到：
$$
\{\boldsymbol{\tilde{w}_1^{(i)}}  ,\boldsymbol{\tilde{w}_2^{(i)}},···,\boldsymbol{\tilde{w}_n^{(i)}}\}  = MH-ATT(\{\boldsymbol{w_1}  ,\boldsymbol{w_2},···,\boldsymbol{w_n}\})\\
\{\boldsymbol{\tilde{e}_1^{(i)}},\boldsymbol{\tilde{e}_2^{(i)}},···,\boldsymbol{\tilde{e}_m^{(i)}}\}  = MH-ATT(\{\boldsymbol{e_1},\boldsymbol{e_2},···,\boldsymbol{e_m}\})
$$
在多头注意力之后，将输出信息进行融合。

对于包含实体信息的 tokens，$e_k=f(w_j)$。
$$
\begin{split}
\boldsymbol{h}_j &= \sigma(\boldsymbol{\tilde{W}}_t^{(i)}\boldsymbol{\tilde{w}}_j^{(i)}+\boldsymbol{\tilde{W}}_e^{(i)}\boldsymbol{\tilde{e}}_k^{(i)}+\boldsymbol{\tilde{b}}^{(i)})  \\
\boldsymbol{w}_j^{(i)} &= \sigma(\boldsymbol{\tilde{W}}_t^{(i)}\boldsymbol{h}_j^{(i)}+\boldsymbol{b}_t^{(i)})\\
\boldsymbol{e}_k^{(i)} &= \sigma(\boldsymbol{\tilde{W}}_e^{(i)}\boldsymbol{h}_j^{(i)}+\boldsymbol{b}_e^{(i)})
 \end{split}
$$
$\sigma$ 是非线性激活函数 GELU，$W$ 和 $b$ 都是训练得到的参数。$\boldsymbol{h}_j$ 是融合 token 和 entity 的隐状态。

对于不包含实体信息的 tokens。
$$
\begin{split}
\boldsymbol{h}_j &= \sigma(\boldsymbol{\tilde{W}}_t^{(i)}\boldsymbol{\tilde{w}}_j^{(i)}+\boldsymbol{\tilde{b}}^{(i)})  \\
\boldsymbol{w}_j^{(i)} &= \sigma(\boldsymbol{\tilde{W}}_t^{(i)}\boldsymbol{h}_j^{(i)}+\boldsymbol{b}_t^{(i)})
 \end{split}
$$
简化 Knowledgeable Encoder 是：
$$
\{\boldsymbol{w}_1^{(i)},\boldsymbol{w}_2^{(i)},···,\boldsymbol{w}_n^{(i)}\},\ \{\boldsymbol{e}_1^{(i)},\boldsymbol{e}_2^{(i)},···,\boldsymbol{e}_m^{(i)}\}\\=Aggregator(\{\boldsymbol{w}_1^{(i-1)},\boldsymbol{w}_2^{(i-1)},···,\boldsymbol{w}_n^{(i-1)}\},\ \{\boldsymbol{e}_1^{(i-1)},\boldsymbol{e}_2^{(i-1)},···,\boldsymbol{e}_m^{(i-1)}\})
$$

### 2.4 预训练与微调  

注入知识的==预训练==

为了将KGs信息表达，作者提出了新一个预训练任务。

对于句子，随机掩盖 token-entity 的 entity，让模型基于对应的 tokens 来预测所有相关的 entites，即
$$
p\left(e_{j} \mid w_{i}\right)=\frac{\exp \left(\text { linear }\left(\boldsymbol{w}_{i}^{o}\right) \cdot \boldsymbol{e}_{j}\right)}{\sum_{k=1}^{m} \exp \left(\text { linear }\left(\boldsymbol{w}_{i}^{o}\right) \cdot \boldsymbol{e}_{k}\right)}
$$
由于实体的数量很多，因此只预测所给的实体序列。

这个过程被叫做 denoising entity auto-encoder (dEA)，降噪实体自动编码器。

$\text { linear }(·)$ 是线性层，损失函数使用交叉熵。

模拟随机噪音：

- 5%的概率，将token-entity对的entity随机替换掉。加入噪音，使得模型在少量token的实体信息错误情况下的鲁棒性。
- 15%概率，掩掉token-entity。
- 其余的正常。

此外，在文本上，ERNIE 采用和 bert 一样的 MLM 和 NSP 任务。总体的预训练损失为 dEA，MLM，NSP的损失之和。



==微调== 

![image-20230306145854503](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230306145854503.png)

> 对于一般任务，输入为 token，和 bert 一样。
>
> 对于 knowledge-driven 任务，
>
> - Entity Typing，实体分类，常常被理解为是一个多标签的多分类问题。
>
>     本文认为带提及标记[ENT]的修正输入序列能够引导 ERNIE 注意结合上下文信息和实体提及信息。
>
> - Relation Classification，该任务要求系统基于上下文对给定实体对的关系标签进行分类。
>
>   [CLS] 这个 token embedding 用于分类，[HD] 是 head entities，  [TL] 是 tail entities。

==Entity mention 实体提及==

**mention就是自然文本中表达实体(entity)的语言片段。**Mention通常被定义为**自然语言文本中对实体的引用**。

如自然文本 “鸟儿打湿了自己的翅膀”，通过ner提取出mention —— “翅膀”。

而一个知识图谱中和“翅膀”有关的entity有如下：

> ["隐形的翅膀（2006年张韶涵演唱歌曲）",  "大空翼",   "今井翼",   "翅膀（香香演唱歌曲）",   "翅膀（飞行生物的飞行器官）",   "翅膀（费翔粉丝名称）", "翅膀（豆豆演唱歌曲）", "翅膀（胡夏演唱歌曲）", "翅膀（羽泉演唱歌曲）", "翅膀（王心凌演唱歌曲）", "翅膀（林俊杰演唱歌曲）", "翅膀（林依晨演唱歌曲）", "翅膀（指人儿乐队(finger family)演唱歌曲）", "翅膀（房震演唱歌曲）", "翅膀（微电影《翅膀》）", "翅膀（宋祉萱演唱的《翅膀》）", "翅膀（天空乐队演唱歌曲）", "翅膀（周迅演唱歌曲）", "翅膀（儿童绘本）", "翅膀（伍佰音乐作品）", "翅膀（伊稀演唱歌曲）", "翅膀（付辛博演唱歌曲）", "翅膀（井东伟演唱歌曲）", "翅膀（chase梦音演唱歌曲）"]

此时1个mention对应了24个entity。

而通过自然文本上下文，将mention“*翅膀*”对应到正确的实体“*翅膀（飞行生物的飞行器官）*”的过程就是Entity Linking 。
