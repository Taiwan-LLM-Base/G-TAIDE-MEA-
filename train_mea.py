from datasets import load_from_disk, load_dataset, Dataset,DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    TrainerCallback
)
from trl import SFTTrainer
import wandb
import torch
import pandas as pd
import json
## For H100 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise 'Boolean value expected.'

def main(dataset_path:str = "../data/dataset/llama3_mea_meeting_train_test_0.1_text",
         output_dir:str = "./models/",
         base_model_path:str = "./llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1_spin-b8.3patch1e0",
         flash_attn:str2bool = True if 'H100' in torch.cuda.get_device_name(0) else False,
         lr:float = 1e-5,
         epoch:int = 5,
         bz:int = 1,
         neftune:int = 0,
         linear_decay:int = 0,
         ):
    run_name = f"{base_model_path.split('/')[-1]}_ft_lr{lr}_epoch{epoch}_bz{bz}"#_neftune{neftune}"
    output_dir = output_dir+run_name
    print('='*50)
    print('output_dir:',output_dir)
    print('dataset_path:',dataset_path)
    print('base_model_path:',base_model_path)
    print('lr:',lr)
    print('epoch:',epoch)
    print('neftune or not:',neftune)
    print('linear_decay or not:',linear_decay)
    print('='*50)
    
    ## load dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    dataset = load_from_disk(dataset_path)
    print(dataset)
    
    ## load base model
    model_config = AutoConfig.from_pretrained(base_model_path)
    device_map = 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        # config=model_config,
        low_cpu_mem_usage=True,
        # use_flash_attention_2=flash_attn,
        device_map=device_map,  # Pass the device_map parameter here
        # torch_dtype=torch.bfloat16
        # torch_dtype=torch.float16,
    )
    wandb.init(project='G-TAIDE_mea',name = run_name) # (要先登入自己的wandb帳號)
    ## 因為使用的是V100，所以本次訓練只能使用FP16。如果有H100，使用BF16效果會更好。
    training_args = TrainingArguments(
        output_dir=output_dir,             # 模型和檢查點的輸出目錄
        num_train_epochs=epoch,            # 訓練的總迴圈數
        per_device_train_batch_size=bz,    # 每個設備的訓練批次大小
        per_device_eval_batch_size=bz,     # 每個設備的評估批次大小
        gradient_accumulation_steps=4,     # 梯度累積的步數，用於處理大批次
        logging_steps=20,                  # 每多少步記錄一次日誌
        learning_rate=lr,                  # 學習率
        warmup_ratio=0.01,                 # 預熱的比例（與總訓練步數相乘得到預熱步數）
        group_by_length=True,              # 按照長度分組，以優化訓練速度和記憶體使用
        lr_scheduler_type="constant",
        save_strategy="epoch",             # 模型儲存策略
        evaluation_strategy='steps',       # 評估策略
        eval_steps=10,                     # 每多少步進行一次評估
        report_to="wandb",                 # 報告到 Weights & Biases 
        run_name=run_name,                 # 訓練任務的名稱
        save_only_model=True,              # 只儲存模型權重，不儲存優化器狀態等
        do_train=True,                     # 是否進行訓練
        do_eval=True,                      # 是否進行評估
        save_total_limit=3,                # 最多儲存多少個模型檢查點
        # max_grad_norm=0.3,                 # 梯度裁剪的最大梯度範數 fp 不支持
        fp16=True,                         # 啟用FP16訓練
    )
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=4096,
        args=training_args,
        # callbacks=[GlobalStepCallback()]
    )
    fine_tuning.train()
    # fine_tuning.save_model(output_dir)
    # print("model has been saved to",output_dir)
    fine_tuning.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    
if __name__ == '__main__':
    import fire
    fire.Fire(main)
    