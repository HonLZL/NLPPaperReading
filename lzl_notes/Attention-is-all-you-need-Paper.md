CNN 和 RNN 不够好，因此本文提出用 attention 结构

## 0 Transformer 结构

单层

<img src="note_images\image-20221123191342472.png" alt="image-20221123191342472" style="zoom:50%;" />



多层 Transformer：原论文中采用了 6 个编码器和 6 个解码器

<img src="note_images\88794918439701.png" alt="88794918439701" style="zoom:67%;" />

​    在多层 Transformer 中，多层编码器先对输入序列进行编码，然后得到最后一个 Encoder 的输出 Memory；解码器先通过 Masked Multi-Head Attention 对输入序列进行编码，然后将输出结果同 Memory 通过 Encoder-Decoder Attention 后得到第 1 层解码器的输出；接着再将第 1 层 Decoder 的输出通过 Masked Multi-Head Attention进行编码，最后再将**编码后的结果同Memory**通过Encoder-Decoder Attention 后得到第 2 层解码器的输出，以此类推得到最后一个 Decoder 的输出。

​    在多层 Transformer的解码过程中，每一个 Decoder 在 Encoder Decoder Attention 中所使用的 Memory 均是同一个。

## 1 Attention

### 1.1 Attention and Self-attention

<img src="note_images\image-20221201205037663.png" alt="image-20221201205037663" style="zoom:50%;" />

注意力机制可以描述为将query 和一系列的 key-value 对映射到某个输出的过程，而**这个输出的向量就是根据 query 和 key 计算得到的权重作用于 value 上的权重和**。
$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_k}})V
$$


### 1.2 Multi-Head Attention

目的：解决多头注意力可以用于克服模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置的问题。

![image-20221201205151330](note_images\image-20221201205151330.png)
$$
MultiHead(Q,K,V)=Concat(head_1,···,head_h)W_O\\
where \ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)\\
W_i^Q\in \mathbb{R}^{d_{model}\times d_k}, \ W_i^K\in \mathbb{R}^{d_{model}\times d_k}, \ W_i^V\in \mathbb{R}^{d_{model}\times d_v}, \ W_i^O\in \mathbb{R}^{hd_v\times d_{model}}
$$
相当于有多个权重矩阵。

作者使用了 h=8 个并行的自注意力模块（8 个头）来构建一个注意力层，并且对于每个自注意力模块都限定了 $d_k = d_v = d_{model} / h = 64$。

### 1.3 Attention Mask

<img src="note_images\image-20221205140033753.png" alt="image-20221205140033753" style="zoom: 67%;" />

在训练过程中的 Decoder 对于每一个样本来说都需要这样一个对称矩阵来掩盖掉当前时刻之后所有位置的信息。目的是为了使得decoder不能看见未来的信息.也就是对于一个序列中的第i个token,解码的时候只能够依靠i时刻之前(包括i)的的输出,而不能依赖于i时刻之后的输出.因此我们要采取一个遮盖的方法(Mask)使得其在计算self-attention的时候只用i个时刻之前的token进行计算,因为Decoder是用来做预测的,而在训练预测能力的时候,我们不能够"提前看答案",因此要将未来的信息给遮盖住。

<img src="note_images\image-20221205142629920.png" alt="image-20221205142629920" style="zoom:50%;" />



### 1.4 Q K V

整个 Transformer 中涉及到自注意力机制的一共有 3 个部分：Encoder 中的 Multi-Head Attention；Decoder 中的 Masked Multi-Head Attention；Encoder 和 Decoder 交互部分的 Multi-Head Attention。

<img src="note_images\image-20221205142451025.png" alt="image-20221205142451025" style="zoom:50%;" />

- 对于 Encoder 中的 Multi-Head Attention 来说，其原始 q、k、v 均是 Encoder的 Token 输入经过 Embedding 后的结果。q、k、v 分别经过一次线性变换（各自乘以一个权重矩阵）后得到了 Q、K、V，然后再进行自注意力运算得到 Encoder 部分的输出结果 Memory。
- 对于 Decoder 中的 Masked Multi-Head Attention 来说，其原始 q、k、v 均是 Decoder 的 Token 输入经过 Embedding 后的结果。q、k、v 分别经过一次线性变换后得到了 Q、K、V，然后再进行自注意力运算得到 Masked Multi-Head Attention 部分的输出结果，即待解码向量。
- 对于 Encoder 和 Decoder 交互部分的 Multi-Head Attention，其原始 q、k、v 分别是上面的待解码向量、Memory 和 Memory。q、k、v 分别经过一次线性变换后得到了 Q、K、V，然后再进行自注意力运算得到 Decoder 部分的输出结果。之所以这样设计也是在模仿传统 Encoder-Decoder 网络模型的解码过程。

### 1.5 Why Self-Attention

1. 降低每层的复杂度
2. 能够并行计算，由所需的最小顺序操作数来衡量。
3. 在网络中计算long-range依赖需要的计算路径长度。即为了学习long-range依赖，信号在网络中必须要经过的路径长度，这个长度越短，模型就越容易学习long-range依赖。

## 2 Add & Norm and Feed Forward

**Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到：

![image-20221123191221498](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221123191221498.png)
$$
LayerNorm(X+MultiHeadAttention(X))\\
LayerNorm(X+FeedForward(X))
$$

**Feed Forward ** 是为了更好的提取特征。通过线性变换，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征。
$$
FFN(x) = max(0, xW1 + b1)W2 + b2
$$

```python
src2 = self.activation(self.linear1(src))
src2 = self.linear2(self.dropout(src2))
src = src + self.dropout2(src2)
src = self.norm2(src)
```



## 3 Embedding

Embedding = Token_embedding + Positional_embedding

```python
src_embed = self.token_embedding(src)  # [src_len, batch_size, embed_dim]
src_embed = self.pos_embedding(src_embed)  # [src_len, batch_size, embed_dim]
```



### 3.1 Token Embedding

是将各个词（或者字）通过一个 Embedding 层映射到低维稠密的向量空间。

可以使用Vocab，建立词典，一个词对应一个索引即 Token。然后对 Token 进行 Embedding。



### 3.2 Positional Embedding

仅用Token Embedding会缺少时序信息，因此考虑将位置信息加入。PE 就是 Positional Embedding 矩阵
$$
PE_{pos, 2i}=sin(pos/10000^{2i/d_{model}}) \\
PE_{pos, 2i+1}=cos(pos/10000^{2i/d_{model}})
$$
<img src="note_images\image-20221201204203055.png" alt="image-20221201204203055" style="zoom:67%;" />



## 4 Self-attention

自注意力机制解决的问题是：神经网络接收的输入是很多大小不一的向量，并且不同向量向量之间有一定的关系，但是实际训练的时候无法充分发挥这些输入之间的关系而导致模型训练结果效果极差。

==QKV机制== 

假设有一个问题：给出一段文本，使用一些关键词对它进行描述！
为了方便统一正确答案，这道题可能预先已经给大家写出来一些关键词作为提示，其中这些给出的提示就可以看作**key**。
而整个文本的信息就相当于**query**，**value**的含义则更加抽象，可比作是你看到这段文本信息后，脑子里复现的答案信息。
假设刚开始大家都不是很聪明，第一次看到这段文本之后脑子里基本上复现的信息**value**就只有提示的这些信息即**key**，因为**key**与**value**基本是相同的。
但是随着对这个问题的深入理解，通过我们的思考脑子里想起来的东西越来越多，并且能够开始对这段文本即**query**，提取关键词信息进行表示，这就是注意力作用的过程，通过这整个过程我们最终脑子里的**value**发生了变化。
根据提示**key**生成了**query**的关键词表示方法，也就是另一种特征表示方法。
刚刚我们说到**key**和**value**一般情况下默认是相同，与**query**是不同的，这种是我们一般的注意力输入形式。
但有一种特殊情况，就是我们**query**与**key**和**value**相同，这种情况我们成为自注意力机制，就如同我们刚刚的例子，使用一般的注意力机制，是使用不同给定文本的关键词表示它。
而自注意力机制，需要用给定文本自身来表述自己，也就是说你需要从源文本中抽取关键词来表述它，相当于对文本自身的一次特征提取。




Q是一组[查询语句](https://www.zhihu.com/search?q=查询语句&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})，V是数据库，里面有若干数据项。对于每一条查询语句，我们期望从数据库中查询出一个数据项（[加权](https://www.zhihu.com/search?q=加权&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})过后的）来。如何查询？这既要考虑每个q本身，又要考虑V中每一个项。如果用K表示一组钥匙，这组钥匙每一把对应V中每一项，代表了V中每一项的某种查询特征，（所以K和V的数量一定是相等的，维度则没有严格限制，做[attention](https://www.zhihu.com/search?q=attention&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})时维度和q一样只是为了在做[点积](https://www.zhihu.com/search?q=点积&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})时方便，不过也存在不用点积的attention）。然后对于每一个Q中的q，我们去求和每一个k的attention，作为对应value的加权系数，并用它来加权数据库V中的每一项，就得到了q期望的查询结果。
所以query是查询语句，value是数据项，key是对应每个数据项的钥匙。名字起得是很生动的。不过和真正的[数据库查询](https://www.zhihu.com/search?q=数据库查询&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})不一样的是，我们不仅考虑了查询语句，还把数据库中所有项都加权作为结果。所以说是全局的。

理论指导实践:1.对于一个文本，我们希望找到某张图片中和文本描述相关的局部图像，怎么办？文本作query(查询），图像做value（数据库）2.对于一个图像，想要找一个文本中和图像所含内容有关的局部文本，如何设计？图像作query，文本作value.3.自注意力（我查我自己）:我们想知道句子中某个词在整个句子中的分量（或者相关文本），怎么设计？句子本身乘以三个矩阵得到Q,K,V，每个词去查整个句子。4.交叉注意力（查别人）:transformer模型的decoder中，由decoder的输入经过变换作为query，由encoder的输出作为key和value（数据库）。value和query来自不同的地方，就是交叉注意力。可以看到key和value一定是代表着同一个东西。即:[Q,(K,V)]。如果用encoder的输出做value，用[decoder](https://www.zhihu.com/search?q=decoder&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2718310467})的输入做key和query 那就完完全全不make sense了。所以从宏观上就可以判断谁该作query，谁该作value和key 。而不是乱设计。

==如何知道什么是重点？== 



| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122142003260.png" alt="image-20221122142003260"  /> | ![image-20221122143006882](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143006882.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221122143151270](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143151270.png) | ![image-20221122143330365](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122143330365.png) |
| 矩阵运算：                                                   |                                                              |
| ![image-20221122145128074](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145128074.png) | ![image-20221122145353567](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145353567.png) |
| ![image-20221122145815255](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122145815255.png) | A'是含有注意力权重的矩阵                                     |



==Multi-head Self-attention== 

| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122150732786.png" alt="image-20221122150732786" style="zoom: 50%;" /><img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122150841991.png" alt="image-20221122150841991" style="zoom:67%;" /> |
| :----------------------------------------------------------- |



==为A'添加位置信息== 

添加位置信息 $e_i$ 

<img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122151707533.png" alt="image-20221122151707533" style="zoom: 50%;" />



==与CNN对比== 

| <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153527654.png" alt="image-20221122153527654" style="zoom:110%;" /> | <img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153552122.png" alt="image-20221122153552122"  /> | ![image-20221122153637812](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153637812.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

==与RNN对比== 

<img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221122153401230.png" alt="image-20221122153401230" style="zoom: 50%;" />





## 5 auto-regressive

<img src="C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20221123153946482.png" alt="image-20221123153946482" style="zoom:50%;" />

