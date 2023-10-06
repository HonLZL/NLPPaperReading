# ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction

## 一 背景

使用纯语言大模型对文档图片进行信息抽取。
但存在两个问题：模态与任务 gap。


LLM 在多模态有 zero-shot 和 few-shot 的能力。




[ID 和 OOD](https://blog.csdn.net/qq_36478718/article/details/122441367?spm=1001.2101.3001.6650.14&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-14-122441367-blog-122437172.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-14-122441367-blog-122437172.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=15)<br>
ID指的是in-distribution数据，也就是我们熟悉的训练数据；OOD指的是out-of-distribution，在不同的领域也可能被叫做outlier或者是anomaly data，说的是与ID分布不一致的数据。其实ID和OOD的界定比较模糊，通常我们是将语意信息相差较大的两个数据集构成ID和OOD。例如，我们在CIFAR-10上训练好了一个图像分类网络，那么对于这个网络来讲，CIFAR-10数据集的图像就是ID数据，而MNIST，或者是SVHN，以及LSUN等数据集就可以看做是OOD。通常一个比较难以回答的问题就是，在CIFAR-100上训练好的网络，那么CIFAR-10对于网络来说是OOD吗？因为二者相似性很高。在我看来，我们构造验证试验的时候，还是需要尽量选取语义信息具有差异性的两个数据集构成ID与OOD。


##  二 方法











