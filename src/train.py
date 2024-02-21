import random
import torch
import numpy as np
import pandas as pd
import sys
import os

from datasets import Dataset
from datasets import DatasetDict

from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Any
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


# 乱数シードの固定
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)


def preprocess_data(
    data: dict[str, Any], tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    """データの前処理"""
    # 記事のトークナイゼーションを行う
    inputs = tokenizer(
        data["odai"], max_length=64, truncation=True
    )
    # 見出しのトークナイゼーションを行う
    # 見出しはトークンIDのみ使用する
    inputs["labels"] = tokenizer(
        data["boke"], max_length=64, truncation=True
    )["input_ids"]
    return inputs


def main():
    # 乱数シードの固定
    seed_everything(seed=42)
    
    # データセットを読み込む
    df_train = pd.read_csv("./data/train.csv")
    df_valid = pd.read_csv("./data/valid.csv")
    
    # Dataset型に変換
    dataset_train = Dataset.from_pandas(df_train)
    dataset_valid = Dataset.from_pandas(df_valid)
    
    # trainとvalidに分割
    dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_valid
    })
    
    # トークナイザを読み込む
    model_name = "megagonlabs/t5-base-japanese-web"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 訓練セットに対して前処理を行う
    train_dataset = dataset["train"].map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )
    # 検証セットに対して前処理を行う
    validation_dataset = dataset["validation"].map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["validation"].column_names,
    )

    # モデルを読み込む
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # collate関数にDataCollatorForSeq2Seqを用いる
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # resultディレクトリのディレクトリ数を取得する
    v_num = len(os.listdir("./ml/t5/result")) + 1
    
    # Trainerに渡す引数を初期化する
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./ml/t5/result/v{v_num}/output_t5_summarization", # 結果の保存フォルダ
        per_device_train_batch_size=16, # 訓練時のバッチサイズ
        per_device_eval_batch_size=16, # 評価時のバッチサイズ
        learning_rate=1e-4, # 学習率
        lr_scheduler_type="linear", # 学習率スケジューラ
        warmup_ratio=0.1, # 学習率のウォームアップ
        num_train_epochs=5, # 訓練エポック数
        evaluation_strategy="epoch", # 評価タイミング
        save_strategy="epoch", # チェックポイントの保存タイミング
        logging_strategy="epoch", # ロギングのタイミング
        load_best_model_at_end=True, # 訓練後に検証セットで最良のモデルをロード
    )

    # Trainerを初期化する
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    # 訓練する
    trainer.train()