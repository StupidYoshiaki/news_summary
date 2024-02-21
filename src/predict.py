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

from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from train import preprocess_data, seed_everything
from tqdm import tqdm



def convert_list_dict_to_dict_list(
    list_dict: dict[str, list]
) -> list[dict[str, list]]:
  """ミニバッチのデータを事例単位のlistに変換"""
  dict_list = []
  # dictのキーのlistを作成する
  keys = list(list_dict.keys())
  for idx in range(len(list_dict[keys[0]])):  # 各事例で処理する
    # dictの各キーからデータを取り出してlistに追加する
    dict_list.append({key: list_dict[key][idx] for key in keys})
  return dict_list

def run_generation(
    dataloader: DataLoader, model: PreTrainedModel
) -> list[dict[str, Any]]:
  """見出しを生成"""
  generations = []
  for batch in tqdm(dataloader):  # 各ミニバッチを処理する
    batch = {k: v.to(model.device) for k, v in batch.items()}
    # 見出しのトークンのIDを生成する
    batch["generated_target_ids"] = model.generate(**batch)
    batch = {k: v.cpu().tolist() for k, v in batch.items()}
    # ミニバッチのデータを事例単位のlistに変換する
    generations += convert_list_dict_to_dict_list(batch)
  return generations
  
def postprocess_title(
    generations: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
):
  """見出しの後処理"""
  results = []
  # 各事例を処理する
  for generation, data in zip(generations, dataset):
    # IDのlistをテキストに変換する
    data["generated_target"] = tokenizer.decode(
      generation["generated_target_ids"],
      skip_special_tokens=True,
    )
    results.append(data)
  return results
  
  
if __name__ == "__main__":
  # 乱数シードの固定
  seed_everything(seed=42)
  
  # データセットを読み込む
  df = pd.read_csv("./data/test.csv")
  
  # Dataset型に変換
  dataset_test = Dataset.from_pandas(df)
  
  # testに分割
  dataset = DatasetDict({
    "test": dataset_test
  })
  
  # resultディレクトリのディレクトリ数を取得する
  v_num = len(os.listdir("./result"))
  
  # トークナイザとモデル, collatorを読み込む
  model_name = f"./ml/t5/result/output_t5_translation/v{v_num}/checkpoint-84155"
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

  # テストセットに対して前処理を行う
  test_dataset = dataset["test"].map(
    preprocess_data,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=dataset["test"].column_names,
  )
  test_dataset = test_dataset.remove_columns(["labels"])
  # ミニバッチの作成にDataLoaderを用いる
  test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
  )
  # 見出しを生成する
  generations = run_generation(test_dataloader, model)
  
  # 見出しテキストを生成する
  results = postprocess_title(generations, dataset["test"], tokenizer)
  
  # ファイルに書き出す
  with open(f"./result/v{v_num}/translation_generated.txt", "w") as f:
    for result in results:
      f.write(f"{result['input']}: {result['generated_target']}\n")