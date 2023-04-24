# TableFormer: Table Structure Understanding with Transformers

## 1 背景

从文档中提取表格很重要

难点

1. 表格的形状和尺寸很多样
2. 缺乏表结构的数据集，标注花销很大
3. 



通常分为两步

1. 找到表格的位置， 本文把这些当作已解决的问题，使用 yolo，mask-RCNN
2. 识别出表格结构

以往的模型都是依赖于文本特征，或者无法提供原始图像中每个表的单元格的边界框。



目标

1. 能够忽略语言，获得任何表格的结构
2. 能够从 PDF 获得尽可能多的信息
3. 建立 表格与 bounding box 的关系



目前存在的两个方法：

1. **Image-to-Text networks** 

   “image-encoder → text-decoder” (IETD)

   “image-encoder → dual decoder” (IEDD)

   二者都需要 OCR，但是 OCR 是一个特别繁琐的过程。

   从可解析 PDF 获得文本信息是本文工作的启发。

2. **Graph Neural networks** 

   可以避免自定义 OCR decoder，但是效果不好。

3. **Hybrid Deep Learning-Rule-Based approach** 混合 深度学习-基于规则的方法

   先检测位置，再识别表结构。但这不是 end-to-end 方法，因此如果遇到不同类型的表格，需要重新编写新的规则。



## 3 数据集

PubTabNet是IBM公司公布的基于图像的表格识别数据集。其包含了568k+表格图片，其标注数据是HTML的表格结构,标记的文本，表格单元的边界框， 该数据集的表格都是PDF截图，清晰度不是很高。

| ![image-20230330211928195](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20230330211928195.png) | <img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230330211913765.png" alt="image-20230330211913765" style="zoom:90%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

