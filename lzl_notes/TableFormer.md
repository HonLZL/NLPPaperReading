# TableFormer: Robust Transformer Modeling for Table-Text Encoding

## 1 背景

以 TaPas 为代表的现有模型容易受到扰动，缺乏结构偏差，单元格联系较差。

例如：

| Song Title   | Length |
| ------------ | ------ |
| Screwed Up   | 5:02   |
| Ghetto Queen | 5:00   |

Question: Of all song lengths, which one is the longest? 

Gold Answer: 5:02 

TAPAS: 5:00 

TAPAS after row order perturbation: 5:02 ，换了个顺序，结果又是正确的了

TABLEFORMER: 5:02

这个例子表明，TaPas 受到顺序的影响

