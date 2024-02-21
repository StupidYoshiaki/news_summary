import random
import torch
import numpy as np
import pandas as pd
import sys
import os

from datasets import Dataset, DatasetDict, load_metric

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
        data["input"], max_length=64, truncation=True
    )
    # 見出しのトークナイゼーションを行う
    # 見出しはトークンIDのみ使用する
    inputs["labels"] = tokenizer(
        data["target"], max_length=64, truncation=True
    )["input_ids"]
    return inputs


# ファインチューニングのmetricとしてBLEUを使用する
def compute_bleu(
    predictions: list[str], references: list[list[str]]
) -> dict:
    """BLUEを算出"""
    # sacreBLEUをロードする
    bleu = load_metric("sacrebleu")
    # 単語列を評価対象に加える
    bleu.add_batch(predictions=predictions, references=references)
    # BLEUを計算する
    results = bleu.compute()
    results["precisions"] = [
        round(p, 2) for p in results["precisions"]
    ]
    return results


def main():
    # 乱数シードの固定
    seed_everything(seed=42)
    
    # GPUが利用可能か確認する
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    
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
    
    # 量子化
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # quant_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

    # モデルを読み込む
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # collate関数にDataCollatorForSeq2Seqを用いる
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # resultディレクトリのディレクトリ数を取得する
    v_num = len(os.listdir("./result/output_t5_translation")) + 1
    
    # Trainerに渡す引数を初期化する
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./result/output_t5_translation/v{v_num}", # 結果の保存フォルダ
        per_device_train_batch_size=8, # 訓練時のバッチサイズ
        per_device_eval_batch_size=8, # 評価時のバッチサイズ
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
        compute_metrics=compute_bleu,
    )

    # 訓練する
    trainer.train()
    
if __name__ == "__main__":
    main()