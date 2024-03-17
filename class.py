import os
os.environ['HTTP_PROXY'] = 'http://sgdo.yulki.codes:24396'
os.environ['HTTPS_PROXY'] = 'http://sgdo.yulki.codes:24396'
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import torch
print(torch.cuda.is_available())
import evaluate

dataset = load_dataset("csv", data_files="/home/jiangyunqi/Code/TangWen/Weibo-classification/response_v0.csv", split="train")
label_list = dataset.unique("label")
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}



tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
datasets = dataset.train_test_split(test_size=0.1)

def process_function(examples):
    examples["label"] = [label2id[label] for label in examples["label"]]
    tokenized_examples = tokenizer(examples["review"], max_length=32, truncation=True, padding="max_length")
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True)
tokenized_datasets

model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-large", num_labels=len(label2id))
model.config.id2label = id2label
model.config.label2id = label2id
print(model.config.id2label)

# 如果网络不太好，也可以使用本地加载的方式
acc_metric = evaluate.load("accuracy")
#f1_metirc = evaluate.load("f1")


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    #f1 = f1_metirc.compute(predictions=predictions, references=labels)
    #acc.update(f1)
    return acc


train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=1,   # 训练时的batch_size
                               gradient_accumulation_steps=32,  # *** 梯度累加 ***
                               gradient_checkpointing=True,     # *** 梯度检查点 ***
                               optim="adafactor",               # *** adafactor优化器 *** 
                               per_device_eval_batch_size=1,    # 验证时的batch_size
                               num_train_epochs=1,              # 训练轮数
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               #metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True,
                               report_to=None)     # 训练完成后加载最优模型
train_args




# *** 参数冻结 *** 
# for name, param in model.bert.named_parameters():
#     param.requires_grad = False

trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()