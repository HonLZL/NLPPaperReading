## 1 huggingface Datasets

### 1.1 自定义数据集加载脚本









## 2 使用 torch 加载自己的数据集

```python
class MyDataset(Dataset):
    def __init__(self, paths_file):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        super(MyDataset, self).__init__()
        with open(paths_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            lines = [line.strip("\n") for line in lines]
        self.image_paths = lines

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）
        2. 对读取到的数据进行数据增强 (数据增强是深度学习中经常用到的，可以提高模型的泛化能力)
        3. 返回数据对 （一般我们要返回 图片，对应的标签） 在这里因为我没有写完整的代码，返回值用 0 代替
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image = transform(image)
        return image

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.image_paths)
```







