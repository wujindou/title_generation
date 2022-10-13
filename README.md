# title_generator

跑通bert4torch 根据内容生成标题的脚本
Question or title generation using bert4torch 

seq2seq.py 根据内容生成问题
seq2seq_train_eval.py 增加在验证集上的效果展示如：rouge-1值和bleu值

自己数据集上效果


|  model   | bleu  |
|  ----  | ----  |
| bart  | 0.162 |
| bert-unilm  | 0.13 |
