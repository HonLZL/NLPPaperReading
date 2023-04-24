[pytorch中文文档](https://www.pytorchtutorial.com/docs/)

## 0 torch

### 0.1 torch.save

保存模型参数和tensor数据





### 0.2 torch.load

从磁盘文件中读取一个通过`torch.save()`保存的对象。`torch.load()` 可通过参数`map_location` 动态地进行内存重映射，使其能从不动设备中读取文件。

参数:

- f – 类文件对象 (返回文件描述符)或一个保存文件名的字符串
- map_location – 一个函数或字典规定如何remap存储位置
- pickle_module – 用于unpickling元数据和对象的模块 (必须匹配序列化文件时的pickle_module )

例子:

```python
torch.load('tensors.pt')
# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Map tensors from GPU 1 to GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
```



```python
import torch

save_torch = torch.Tensor([[1, 2, 3, 4],
                           [2, 34, 5, 6]])
print(save_torch)
torch.save(save_torch, 'test_save_tensor.pt') # 保存
load_torch = torch.load('test_save_tensor.pt') # 读取
print(load_torch)
load_torch == save_torch
```

通过 这两个操作，将处理好的数据直接保存，能在多次测试期间节省处理数据的时间。





## 1 torch.nn

### 1.1 nn.embedding

一个保存了固定字典和大小的简单查找表。

这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

```python
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
参数：

    num_embeddings (int) - 嵌入字典的大小
    embedding_dim (int) - 每个嵌入向量的大小
    padding_idx (int, optional) - 如果提供的话，输出遇到此下标时用零填充
    max_norm (float, optional) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
    norm_type (float, optional) - 对于max_norm选项计算p范数时的p
    scale_grad_by_freq (boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度

变量：

    weight (Tensor) -形状为(num_embeddings, embedding_dim)的模块中可学习的权值

形状：

    输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
    输出： (N, W, embedding_dim)
```



```python
self.embedding = nn.Embedding(vocab_size, emb_size)
```

self.embedding本质上就是一个查找表，词作为下标去索引。

例子：

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

embedding = nn.Embedding(10, 3)
input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
# tensor([[1, 2, 4, 5],
#         [4, 3, 2, 9]])
embedding(input)
#tensor([[[-0.1944, -0.5118,  0.2785],
#         [ 0.2809,  0.4442, -0.1756],
#         [ 0.6525,  1.7653, -0.7876],
#         [ 2.1310,  1.4708, -0.7405]],
#
#        [[ 0.6525,  1.7653, -0.7876],
#         [-1.5050,  0.3106,  1.0984],
#         [ 0.2809,  0.4442, -0.1756],
#         [ 0.0257,  0.3683, -1.0249]]], grad_fn=<EmbeddingBackward0>)
```



### 1.2 nn.Dropout

```python
class torch.nn.Dropout(p=0.5, inplace=False)
```

**作用：**

  **Dropout的是为了防止过拟合而设置**

  随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。

**参数：**

- **p** - 将元素置0的概率。默认值：0.5
- **in-place** - 若设置为True，会在原地执行操作。默认值：False

**形状：**

- **输入：** 任意。输入可以为任意形状。
- **输出：** 相同。输出和输入形状相同。

```python
import torch
import torch.nn as nn

a = torch.randn(4, 4)
print(a)
"""
tensor([[ 1.2615, -0.6423, -0.4142,  1.2982],
        [ 0.2615,  1.3260, -1.1333, -1.6835],
        [ 0.0370, -1.0904,  0.5964, -0.1530],
        [ 1.1799, -0.3718,  1.7287, -1.5651]])
"""
dropout = nn.Dropout()
b = dropout(a)
print(b)
"""
tensor([[ 2.5230, -0.0000, -0.0000,  2.5964],
        [ 0.0000,  0.0000, -0.0000, -0.0000],
        [ 0.0000, -0.0000,  1.1928, -0.3060],
        [ 0.0000, -0.7436,  0.0000, -3.1303]])
"""
```



### 1.3 nn.Liner

```python
class torch.nn.Linear(in_features, out_features, bias=True)
```

线性层，也叫全连接层。对输入数据做线性变换：$Y_{n\times o}=X_{n\times i}W_{i\times o}+b$ ,W 是模型要学习的参数

**参数：**

- **in_features** - 每个输入样本的大小
- **out_features** - 每个输出样本的大小
- **bias** - 若设置为False，这层不会学习偏置。默认值：True

**形状：**

- **输入:** $(N,in\_features)$
- **输出：**$(N,out\_features)$

```python
m = nn.Linear(20, 30)
input = torch.autograd.Variable(torch.randn(128, 20))
output = m(input)
print(input.size())
# torch.Size([128, 20])
print(output.size())
# torch.Size([128, 30])
```





### 1.2 nn.LayerNorm





### 1.3 nn.functional

 ```python
 import torch.nn.functional as F
 ```

#### 1.3.1 F.softmax

$$
Softmax(x_i)=\frac{exp(x_i)}{\sum_jexp(x_j)}
$$

```python
F.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

- **input** (*Tensor*) – input
- **dim** (*int*) – A dimension along which softmax will be computed.
- **dtype** (*torch.dtype*, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to *dtype* before the operation is performed. This is useful for preventing data type overflows. Default: None.

其输入值是一个向量，向量中元素为任意实数的评分值，softmax 函数对其进行压缩，输出一个向量，其中每个元素值在0到1之间，且所有元素之和为1



## 2 torch.utils

### 2.1 data

#### 2.1.1 data.Dataset

Dataset：真正的“数据集”，它的作用是：**只要告诉它数据在哪里(初始化)，就可以像使用iterator一样去拿到数据**，继承该类后，需要重载`__len__()`以及`__getitem__`。当有了数据集之后，才能去 loader。

这个类一般用于 继承。

```python
from torch.utils.data.dataset import Dataset
class MyDataset(Dataset):
    def __init__(self, img_dir, anno_file, imgsz=(640, 640)):
        self.img_dir = img_dir
        self.anno_file = anno_file
        self.imgsz = imgsz
        self.img_namelst = os.listdir(self.img_dir)

    # need to overload
    def __len__(self):
        return len(self.img_namelst)

    # need to overload
    def __getitem__(self, idx):
        with open(self.anno_file, 'r') as f:
            label = f.readline().strip()
        img = cv2.imread(os.path.join(img_dir, self.img_namelst[idx]))
        img = cv2.resize(img, self.imgsz)
        return img, label


dataset = MyDataset(img_dir, anno_file)
```



#### 2.1.2 DataLoader

```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
```

对上面的 数据集 进行 load：

```python
dataloader = DataLoader(dataset=dataset, batch_size=2)
# display
for img_batch, label_batch in dataloader:
    img_batch = img_batch.numpy()
    print(img_batch.shape)
    # img = np.concatenate(img_batch, axis=0)
    if img_batch.shape[0] == 2:
        img = np.hstack((img_batch[0], img_batch[1]))
    else:
        img = np.squeeze(img_batch, axis=0)  # 最后一张图时，删除第一个维度
    print(img.shape)
    cv2.imshow(label_batch[0], img)
    cv2.waitKey(0)
```

**参数：**

- **dataset** (*Dataset*) – 加载数据的数据集。
- **batch_size** (*int*, optional) – 每个batch加载多少个样本(默认: 1)。
- **shuffle** (*bool*, optional) – 设置为`True`时会在每个epoch重新打乱数据(默认: False).
- **sampler** (*Sampler*, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略`shuffle`参数。
- **num_workers** (*int*, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
- **collate_fn** (*callable*, optional) –
- **pin_memory** (*bool*, optional) –
- **drop_last** (*bool*, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)



## 3 Tensor 对象

### 3.1 view()

返回一个有相同数据但 size 不同的tensor。 返回的tensor必须有与原tensor相同的数据和相同数目的元素，但可以有不同的尺寸。一个tensor必须是连续的`contiguous()`才能被查看。

```python
x = torch.randn(3, 4)
# tensor([[ 0.1124, -0.2316,  0.1721, -1.1407],
#         [ 1.3223,  0.3190,  1.5236, -0.6364],
#         [ 0.3575, -0.2414, -1.2291,  1.5323]])

y = x.view(2, 6)
# tensor([[ 0.1124, -0.2316,  0.1721, -1.1407,  1.3223,  0.3190],
#         [ 1.5236, -0.6364,  0.3575, -0.2414, -1.2291,  1.5323]])

z = x.view(-1, 6)  # -1 是自动计算第一个，所以和 y 一样
# tensor([[ 0.1124, -0.2316,  0.1721, -1.1407,  1.3223,  0.3190],
#         [ 1.5236, -0.6364,  0.3575, -0.2414, -1.2291,  1.5323]])
```



### 3.2 argmax() 和 argmin()



```python
x = torch.randn(3, 4)
# tensor([[ 1.9084,  0.8511,  0.7899,  1.1031],
#         [-0.1445, -0.4266, -0.7124, -1.2067],
#         [ 0.6999,  0.8786,  0.6875, -0.4608]])

x.argmax(1)  # 求每行最大值的下标
# tensor([0, 0, 1])

x.argmax(0)   # 求每列最大值的下标
# tensor([0, 2, 0, 0])

x.argmin(1)  # 求每行最小值的下标
# tensor([2, 3, 3])

x.argmin(0)   # 求每列最小值的下标
# tensor([1, 1, 1, 1])
```





















































```python
torch.argmax(input, dim, keepdim=False) → LongTensor
```







