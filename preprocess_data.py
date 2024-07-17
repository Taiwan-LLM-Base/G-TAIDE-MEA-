import pandas as pd
from datasets import Dataset, load_dataset
## 直接讀取 jsonl 檔案成 dataset
dataset = load_dataset("json", data_files="./開會通知單.jsonl")['train']
## 套用 llama3 格式
questions_text = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant, particularly skilled at handling official documents.<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>".format(prompt = dataset["prompt"][i], answer = dataset["response"][i]) for i in range(len(dataset))]
## 存成 text 的形式，for huggingface trainer
data = pd.DataFrame({"text":questions_text})
dataset = Dataset.from_pandas(data)
dataset.shuffle(seed=42)
dataset=dataset.train_test_split(test_size=0.1)
dataset.save_to_disk("../data/dataset/"+"llama3_mea_meeting_train_test_0.1_text")




## Llama3 special token format
'''
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{sys prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer}
<|eot_id|>
'''# 多輪不加 <|end_of_text|>