# PAPERS

## 一 基础的预训练模型

### 1 transformer

​	Transformer是由Google于2017年提出的一种自注意力机制（self-attention）的神经网络模型，在自然语言处理领域中得到了广泛应用和成功。

​	与传统的递归和卷积模型不同，Transformer使用自注意力机制来对输入进行编码和解码。它能够在一次前向传递中处理整个序列，而不需要像递归模型那样依次处理每个单词或者像卷积模型那样只考虑局部上下文信息。	这使得Transformer在捕捉长距离依赖关系方面有着更好的表现，同时还可以高效地并行处理。因此，Transformer已成为自然语言处理领域的主要框架之一，并被广泛应用于文本分类、情感分析、机器翻译、文本生成等多个任务。



[notebook](Attention-is-all-you-need-Paper.md)

### 2 Bert

​	BERT是由Google在2018年提出的一种预训练语言模型，全称为Bidirectional Encoder Representations from Transformers。

BERT通过使用大规模未标记文本来进行预训练，使得它可以更好地理解自然语言中单词之间的关系，从而在各种自然语言处理任务中获得优秀的表现。与传统的基于卷积或循环神经网络的模型相比，BERT具有以下特点：

1. 双向编码器：BERT采用双向编码器模型，使得每个单词能够考虑上下文信息，从而在语义理解方面取得了重大进展。
2. 预训练过程：BERT使用了一个庞大的未标记文本语料库来进行预训练，并将其转化为有监督学习任务中的特定任务。
3. 适应性微调：BERT还可以很容易地通过微调的方式进行适应不同的自然语言处理任务和数据集。

​	BERT已经成为自然语言处理领域的标杆模型，被广泛应用于文本分类、情感分析、问答系统、命名实体识别等多个任务，在相关评测中都取得了卓越的表现。



[notebook](BERT.md)



### 3 Bart

​	BART是Facebook于2020年提出的一种基于Transformer架构的序列到序列预训练模型。BART全称为Bidirectional and Auto-Regressive Transformer，即双向自回归 transformer。

​	相较于传统的自回归语言生成模型如GPT系列，BART通过在训练过程中加入了对序列的反向重构和随机缺失等操作，从而更好地利用了无标签数据，进一步提高了模型的泛化能力和效果。

​	BART不仅适用于文本生成任务，比如摘要、翻译和对话生成等，还可以应用于问答、文本分类等多个自然语言处理任务上。在各种评测中都取得了一些优秀的成果。

[notebook](BART.md)



### 4 GPTs

[notebooks](GPTs.md)

**GPT1**

GPT全称为Generative Pre-training Transformer，是由OpenAI提出的一种预训练语言模型。它采用了Transformer架构，并使用基于自回归（autoregressive）的方式进行训练。

在训练过程中，GPT通过对大规模无标注文本进行预训练，使得模型可以学习到单词之间的关系和上下文信息，并能够根据输入文本生成连贯的、有意义的输出文本。由于其卓越的表现，GPT被广泛应用于文本生成、机器翻译、对话系统等多个领域，具有以下优点：

1. 适应各种任务：GPT可以通过微调的方式很容易地适应不同的自然语言处理任务和数据集。
2. 提供连续的文本生成：GPT可以基于上下文生成连贯的、有意义的输出文本，并且可以生成任意长度的序列。
3. 多领域应用：GPT不仅在纯文本生成方面表现优异，而且还可应用于机器翻译、问答系统、情感分析、推荐系统等多个领域。



**GPT2**

相比于GPT-1，GPT-2在训练数据集、训练时间和模型参数上都有所增加，使得GPT-2能够更好地理解自然语言，并生成更加准确、连贯的文本。与GPT-1相比，GPT-2的主要改进包括：

1. 更大的训练数据集：GPT-2的训练数据集比GPT-1的大约十倍。
2. 更长的训练序列：GPT-2可以处理长度达到2048个单词的序列，在捕捉长距离依赖关系方面表现更出色。
3. 更多的模型参数：GPT-2拥有超过10亿个参数，比GPT-1的4倍还多



### 5 RoBERTa

​	RoBERTa是一种使用大规模未标记语料库进行预训练的语言模型。相比于其前身BERT，RoBERTa的优势在于它使用了更多、更多样化的数据来进行预训练，以及使用了更长的训练时间和更大的批次大小。

​	这些改进使得RoBERTa在各种自然语言处理任务上取得了更好的性能表现，包括问答、文本分类、命名实体识别等。此外，RoBERTa还提供了更多的预训练参数选项，可以根据具体任务需求进行调整，从而更好地解决特定领域下的自然语言处理问题。

[notebook](RoBERTa.md)





### 6 Swin Transformer

​	Swin Transformer 是一种轻量级的高效图像识别模型，它采用了跨阶段连接（cross-stage  connection）和窗口交换注意力（window-based multi-head  self-attention）等新颖的机制，在保持准确率的同时大幅降低了参数数量和计算复杂度。这使得 Swin Transformer  在计算资源有限的环境下表现优异，并在多个视觉任务上取得了 state-of-the-art 的成果。

[notebook](swin transformer.md)



### 7 DeBERTa









## 10 Vit



: An image is worth 16x16 words: Transformers for image recognition at scale









# 二 文档信息提取

## 1 QKVFormer

A Question-Answering Approach to Key Value Pair Extraction from Form-like Document Images

专注于提取 Key-Value pairs。首先用 encoder 提取出所有的 Keys，然后将其作为 Query，喂进 decoder，去预测 key 对应的 value。

[notebook](KVPFormer-A Question-Answering Approach to Key Value Pair Extraction from Form-like Document Images.md)





## 2 LayoutLMs

**LayoutLM**

LayoutLM: Pre-training of Text and Layout for Document Image Understanding



**LayoutLMv2**

LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding



**LayoutXLM**

LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding



**LayoutLMv3**

LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking



[notebooks](LayoutLMs.md)



## 3 GeoLayoutLM

GeoLayoutLM: Geometric Pre-training for Visual Information Extraction
