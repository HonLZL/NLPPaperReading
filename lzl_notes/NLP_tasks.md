# 基于预训练模型的 NLP 数据集与任务 



## 1 WikiSQL

> dev.db          验证集
>
> dev.jsonl
>
> dev.tables.jsonl
>
> test.db
>
> test.jsonl
>
> test.tables.jsonl
>
> train.db
>
> train.jsonl
>
> train.tables.jsonl

行列都从 0 开始。

具体内容：

- .db 是数据库文件，里面是多个表，取出一个表可视化

  可以用以下代码读取 .db 文件

  ```python
  from sqlite3 import Error
  import pandas as pd
  import sqlite3
  
  try:
      conn = sqlite3.connect("data/dev.db")
  except Error as e:
      print(e)
  
  # Now in order to read in pandas dataframe we need to know table name
  cursor = conn.cursor()
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
  # print(f"Table Name : {cursor.fetchall()}")
  Table_Names = cursor.fetchall()
  Table_Name = Table_Names[0][0]
  
  df = pd.read_sql_query("SELECT * FROM {}".format(Table_Name), conn)
  print(df)
  conn.close()
  ```

  可视化结果如下

  由 .tables.jsonl 内容可知，列名为：

  "Player",	   "No.",	"Nationality",	"Position",	  "Years in Toronto",	"School/Club Team"

  |      | col0          | col1   | col2          | col3           | col4         | col5           |
  | ---- | ------------- | ------ | ------------- | -------------- | ------------ | -------------- |
  | 0    | antonio lang  | 21     | united states | guard-forward  | 1999-2000    | duke           |
  | 1    | voshon lenard | 2      | united states | guard          | 2002-03      | minnesota      |
  | 2    | martin lewis  | 32, 44 | united states | guard-forward  | 1996-97      | butler cc (ks) |
  | 3    | brad lohaus   | 33     | united states | forward-center | 1996         | iowa           |
  | 4    | art long      | 42     | united states | forward-center | 2002-03      | cincinnati     |
  | 5    | john long     | 25     | united states | guard          | 1996-97      | detroit        |
  | 6    | kyle lowry    | 3      | united states | guard          | 2012-present | villanova      |

- .jsonl  里面每行是一个json格式字典，每行表示一个问题。每个表可能有多个问题，即有多行 json 字典对应一个 table_id

  WikiSQL数据集进一步把SQL语句结构化（简化），分成了`conds`，`sel`，`agg`三个部分。所以 **`sql`** 就是label.

  - `sel`是查询目标列，其值是表格中对应列的序号；
  - `agg`的值是聚合操作的编号，可能出现的聚合操作有`['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']`共6种；
  - `conds`是筛选条件，可以有多个。每个条件用一个三元组`(column_index, operator_index, condition)`表示，可能的`operator_index`共有`['=', '>', '<', 'OP']`四种，`condition`是操作的目标值，这是不能用分类解决的目标。

  ```sql
  # 选择某列
  SELECT sel_name FROM table_id 
  # 应用聚合函数
  SELECT MAX(sel_name) FROM table_id 
  SELECT COUNT(sel_name) FROM table_id
  # 应用筛选条件 
  SELECT SUM(sel_name) FROM table_id WHERE column_index_name conds[operator_index] condition_
  # 例子
  SELECT SUM(Player) FROM 1-10015132-11 WHERE No.   >   20
  # 下面第一json
  
  SELECT Position FROM 1-10015132-11 WHERE School/Club Team =  "Butler CC (KS)"
  
  SELECT COUNT(Player) FROM 1-10015132-11 WHERE Position =  "guard"
  ```

  

  ```json
  {
      "phase": 1,
   	"table_id": "1-10015132-11",
      "question": "What position does the player who played for butler cc (ks) play?",
      "sql": {
          "sel": 3,
          "conds": [[5, 0, "Butler CC (KS)"]]
          "agg": 0
      }
  }
  {
      "phase": 1,
      "table_id": "1-10015132-11",
      "question": "How many schools did player number 3 play at?",
      "sql": {
          "sel": 5,
          "conds": [[1, 0, "3"]],
          "agg": 3
      }
  }
  
  ```

- .tables.jsonl，每一行为一个表格数据，用 json 的形式将表格重现

  ```json
  {
  	"header": [
  		"Player",
  		"No.",
  		"Nationality",
  		"Position",
  		"Years in Toronto",
  		"School/Club Team"
  	],
  	"page_title": "Toronto Raptors all-time roster",
  	"types": [
  		"text",
  		"text",
  		"text",
  		"text",
  		"text",
  		"text"
  	],
  	"id": "1-10015132-11", 
  	"section_title": "L",
  	"caption": "L",
  	"rows": [
  		[
  			"Antonio Lang",
  			"21",
  			"United States",
  			"Guard-Forward",
  			"1999-2000",
  			"Duke"
  		],
  		[
  			"Voshon Lenard",
  			"2",
  			"United States",
  			"Guard",
  			"2002-03",
  			"Minnesota"
  		],
  		[
  			"Martin Lewis",
  			"32, 44",
  			"United States",
  			"Guard-Forward",
  			"1996-97",
  			"Butler CC (KS)"
  		],
  		[
  			"Brad Lohaus",
  			"33",
  			"United States",
  			"Forward-Center",
  			"1996",
  			"Iowa"
  		],
  		[
  			"Art Long",
  			"42",
  			"United States",
  			"Forward-Center",
  			"2002-03",
  			"Cincinnati"
  		],
  		[
  			"John Long",
  			"25",
  			"United States",
  			"Guard",
  			"1996-97",
  			"Detroit"
  		],
  		[
  			"Kyle Lowry",
  			"3",
  			"United States",
  			"Guard",
  			"2012-Present",
  			"Villanova"
  		]
  	],
  	"name": "table_10015132_11"
  }
  ```

  

## 2 CORD 

| <img src="C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\output.png" alt="output"  /> | ![output](C:\Users\lzl\Desktop\NLPPaperReading\lzl_notes\note_images\output2345.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

```json
{
      "words": [
        {
          "quad": {
            "x2": 182,
            "y3": 700,
            "x3": 182,
            "y4": 700,
            "x1": 148,
            "y1": 666,
            "x4": 148,
            "y2": 666
          },
          "is_key": 0,
          "row_id": 2156639,
          "text": "1X"
        }
      ],
      "category": "menu.cnt",
      "group_id": 3,
      "sub_group_id": 0
}
```



## 3 FUNSD

<img src="note_images\0060308251.png" alt="0060308251" style="zoom:67%;" />



标注是 .json 文件，包含

文本的 box，文本内单词的 box

文本的 label，linking

## 4 XFND

同上，但是有多种语言，图片也更清晰了。



## A 文本分类任务

## B 文本蕴含任务

## C SWAG选择任务

## D **the GLUE benchmark** 

是什么

GLUE（General Language Understanding Evaluation），GLUE包含九项NLU任务，语言均为英语。GLUE九项任务涉及到自然语言推断、文本蕴含、情感分析、语义相似等多个任务。像BERT、XLNet、RoBERTa、ERINE、T5等知名模型都会在此基准上进行测试。目前，大家要把预测结果上传到官方的网站上，官方会给出测试的结果。

