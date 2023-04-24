## 1 Layout-v3

### 1.1 安装环境

```shell
pip install gradio

huggingface-cli login  # 登录 huggingface
```





### 1.2 数据集

**SOIRE** 



```python
from datasets import load_dataset
# this dataset uses the new Image feature :)
dataset = load_dataset("darentang/generated")
```

![image-20230403204852128](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\image-20230403204852128.png)

下载失败，使用报错信息中的 url，将 https 改成 http 手动下载，压缩包解压进

D:\Desktop\ai\huggingface_data\datasets\darentang___generated\sroie\1.0.0