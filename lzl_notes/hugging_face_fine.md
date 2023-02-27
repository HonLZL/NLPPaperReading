# 基于 Hugging Face 对 bert 微调进行文本分类

### 1 安装 hugging face 

```shell
pip install transformers
```

### 2 加载模型

以 [bert-base-uncased](https://huggingface.co/bert-base-uncased) 为例，用英文数据集进行预训练的。

bert-base-uncased    110M 	 English



##### 2.1 hugging face 网站下载

[bert-base-uncased 文件列表](https://huggingface.co/bert-base-uncased/tree/main)

![image-20230222200358539](note_images\Attention-is-all-you-need-Paper.md)

下载里面的 config.json，pytorch_model.bin，vocab.txt，分词还需要 tokenizer.json，tokenizer_config.json。下载完成后，将其复制到自己的项目的一个文件夹里面。

![image-20230223144930128](note_images\image-20230223144930128.png)

使用时，传入文件夹路径：

```python
model = BertModel.from_pretrained('./bert_model')
tokenizer = BertTokenizer.from_pretrained('./bert_model')
```



##### 2.2 用代码下载模型

```python
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

代码下载可能会报错

![image-20230222200636360](C:\Users\lzl\AppData\Roaming\Typora\typora-user-images\image-20230222200636360.png)

下载完成后，模型等文件默认会保存到下面这个路径，还可以通过修改 **HF_HOME** 来自定义下载的目录，具体修改办法是在环境变量→系统变量里新建变量 **HF_HOME**，变量值设置为新位置。

```
C:\Users\lzl\.cache
```



### 3 加载hugging face里的数据



安装 pip install datasets



代码下载数据集，默认保存位置同上。

```python
dataset = load_dataset("beans", split="train")
```

我们使用MRPC数据集中的[GLUE 基准测试数据集](https://gluebenchmark.com/)

```python
datasets = load_dataset("glue", "mrpc")
```

数据集以 apache arrow 文件存储在硬盘。？

读取后数据集格式如下。

| ![image-20230223152834940](note_images\image-20230223152834940.png) | <img src="note_images\image-20230223152847878.png" alt="image-20230223152847878" style="zoom:80%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



查看一条数据

```json
raw_train_dataset = raw_datasets["train"][0]
结果为：
{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
    'label': 1,
    'idx': 0
}
```



### 4 Tokenizer （标记器或者分词器）

将文本转换为模型可以处理的数据，比如将文字输入转换为数字。

```python
tokenizer = AutoTokenizer.from_pretrained('bert_model')
inputs = tokenizer("This is the first sentence.", "This is the second one.")
```

inputs 结果为如下

```json
{
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

input_ids 表示 token 转换后的 数字，这里用 id 表示；

token_type_ids 表示那个是第一句，哪个是第二句，等等；

attention_mask 表示对句子里的哪些 token 做 self-attention，默认对整个句子。



将文本转换为数字，将数字转换为文本：

```python
ids = tokenizer.convert_tokens_to_ids(tokens)
tokens = tokenizer.decode(ids)
```



### 5 处理数据



[datasets.map()](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map)

这个方法的工作原理是在数据集的每个元素上应用一个函数。

```python
def tokenize_function(example):
    tokenizer = AutoTokenizer.from_pretrained('bert_model')
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")

print(raw_datasets["train"][0])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print(tokenized_datasets["train"][0])
```



二者分别为：

```json
{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
    'label': 1,
    'idx': 0
}

{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',

    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
    'label': 1,
    'idx': 0,
    'input_ids': [101, 2572, 3217, 5831, 5496, 2010, 2567, 1010, 3183, 2002, 2170, 1000, 1996, 7409, 1000, 1010, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102, 7727, 2000, 2032, 2004, 2069, 1000, 1996, 7409, 1000, 1010, 2572, 3217, 5831, 5496, 2010, 2567, 1997, 9969, 4487, 23809, 3436, 2010, 3350, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

```

结果是给每一项添加了 tokenizer 信息。



对每个 batch 进行 padding：

```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = data_collator(samples)
s = {k: v.shape for k, v in batch.items()}
print(s)
# {'input_ids': torch.Size([8, 67]), 'token_type_ids': torch.Size([8, 67]), 'attention_mask': torch.Size([8, 67]), 'labels': torch.Size([8])}

```

当你实例化它时，需要一个 tokenizer (用来知道使用哪个词来填充（默认用 0 填充），以及模型期望填充在左边还是右边（默认填充右边）)。可以看到，长度都是 67。



### 6 使用 Trainer Api 进行微调

1 在我们定义我们的 **Trainer** 之前首先要定义一个 **TrainingArguments** 类，它将包含 **Trainer** 用于训练和评估的所有超参数。需要提供保存训练模型的目录,例如 test-trainer，训练时，将会把 checkpoint 等信息保存至该文件夹。

```python
training_args = TrainingArguments("test-trainer")
```

2 定义模型

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

3 定义 Trainer 类

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

直接以 trainer.train() 开始训练的话，每个 epoch 只会有 loss 信息，下面将引入评估部分。



4 评估

使用 trainer.predict()进行预测。

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

**trainer.predict()** 的输出结果是具有三个字段的命名元组： **predictions** , **label_ids** ， 和 **metrics** 。 **metrics** 字段将只包含传递的数据集的loss，以及一些运行时间（预测所需的总时间和平均时间）。如果我们定义了自己的 **compute_metrics()** 函数并将其传递给 **Trainer** ，该字段还将包含**compute_metrics()**的结果。

```python
metrics=  {
    'test_loss': 0.9445633292198181,
    'test_accuracy': 0.3161764705882353,
    'test_f1': 0.0,
    'test_runtime': 7.2147,
    'test_samples_per_second': 56.551,
    'test_steps_per_second': 7.069
}
```

定义了自己的 **compute_metrics()** 函数并将其传递给 **Trainer** ，该字段还将包含**compute_metrics()** 的结果。



使用 **evaluate.load()** 函数，返回的对象有一个 **compute()**方法我们可以用来进行评估计算的方法。

```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)

import evaluate
metric = evaluate.load("glue", "mrpc")
eval_res = metric.compute(predictions=preds, references=predictions.label_ids)
```

将以上加入 Trainer

```python
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```





5 综上，完成第一次训练，完整代码为：

```python
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, \
    Trainer

my_model_path = 'bert_model'

raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(my_model_path)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(my_model_path, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

由于加入评估部分，它将在训练loss之外，还会输出每个 epoch 结束时的验证loss和指标。

### 7 使用纯 Pytorch 完成微调

首先要修改数据集格式，使用 tokenized_datasets 。具体来说，我们需要:

- 删除与模型不期望的值相对应的列（如`sentence1`和`sentence2`列）。
- 将列名`label`重命名为`labels`（因为模型期望参数是`labels`）。
- 设置数据集的格式，使其返回 PyTorch 张量而不是列表。

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
```

使用 Pytorch 的 Dataloader 定义数据集，同样也可以使用 Dataloader 自定义数据集。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

用数据集的一个 batch 测试：

```python
model = AutoModelForSequenceClassification.from_pretrained(my_model_path, num_labels=2)
batch = next(iter(train_dataloader))  # 从dataloader中取出一个batch
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

结果：tensor(0.7016, grad_fn=<NllLossBackward0>) torch.Size([8, 2])



**定义一个训练规则**，优化器使用 AdamW，默认使用的学习率调度器只是从最大值 (5e-5) 到 0 的线性衰减。

```python
from transformers import AdamW, get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,    # 优化器
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```



**定义 device**，并应用到 model 上。

```python
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```



**开始训练**，定义一个进度条。

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```



**保存模型**，使用 pytorch

```python
#保存模型参数
torch.save(model.state_dict(), "new_classify.pth")
#加载模型参数
model.load_state_dict(torch.load("new_classify.pth"))
```



**评估方法**，使用 evaluate，可以写在训练过程里。

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

result = metric.compute()
print(result)
```

输出: {'accuracy': 0.6838235294117647, 'f1': 0.8122270742358079}



**完整的训练过程**：

```python
import evaluate
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler, \
    AdamW

my_model_path = 'bert_model'
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(my_model_path)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(my_model_path, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

progress_bar = tqdm(range(num_training_steps))
model.train()

best_acc = 0
for epoch in range(num_epochs):
    loss_sum = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_sum += loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    if epoch % 1 == 0:
        metric = evaluate.load("glue", "mrpc")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        result = metric.compute()
        print(loss_sum, result)
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            torch.save(model.state_dict(), "best_acc_classify.pth")
```

tensor(251.7992, device='cuda:0', grad_fn=<AddBackward0>) {'accuracy': 0.8161764705882353, 'f1': 0.8803827751196173}

tensor(115.9130, device='cuda:0', grad_fn=<AddBackward0>) {'accuracy': 0.8725490196078431, 'f1': 0.9068100358422939}

tensor(18.6904, device='cuda:0', grad_fn=<AddBackward0>) {'accuracy': 0.8627450980392157, 'f1': 0.904109589041096}



加载微调后的模型

```python
model = AutoModelForSequenceClassification.from_pretrained(my_model_path, num_labels=2)
model.load_state_dict(torch.load("best_acc_classify.pth"))
```

经过测试，加载后的模型评估结果与训练过程的评估结果相符。



### 8 使用Hugging Face 的 Accelerate 加速 Pytorch 训练 

使用 Accelerate 库可以在多个 GPU 或 TPU 上启用分布式训练。

安装 

```python
pip install accelerate
```

与上面训练代码修改内容如下

```python
# 初始化 accelerator
accelerator = Accelerator()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# 替换为
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)


# 删去 训练 和 评估的这行代码
batch = {k: v.to(device) for k, v in batch.items()}


loss.backward()
# 替换为
accelerator.backward(loss)
```



修改完成后，终端输入下面的命令，然后将配置训练的 device 信息。

```shell
accelerate config
```

<img src="note_images\image-20230226143527493.png" alt="image-20230226143527493" style="zoom:67%;" />

完成 accelerate config 后输入下面命令开始训练：

```shell
accelerate launch train.py
```



修改为 accelerate 加速的完整代码为

```python
from accelerate import Accelerator
import evaluate
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler, \
    AdamW

accelerator = Accelerator()

my_model_path = 'bert_model'
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(my_model_path)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(my_model_path, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

progress_bar = tqdm(range(num_training_steps))
model.train()

best_acc = 0
for epoch in range(num_epochs):
    loss_sum = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss_sum += loss

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    if epoch % 1 == 0:
        metric = evaluate.load("glue", "mrpc")
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        result = metric.compute()
        print(loss_sum, result)
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            torch.save(model.state_dict(), "best_acc_classify.pth")
```



一个GPU下的测试：

一个 epoch 下训练所需要的时间

accelerate 加速前：137.6758098602295

加速后：137.5935959815979s

无区别。。。