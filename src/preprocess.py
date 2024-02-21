import pandas as pd
from sklearn.model_selection import train_test_split

# dataフォルダの中にあるcsvファイルを読み込む
df_1 = pd.read_csv("./data/easy_japanese.csv")
df_2 = pd.read_csv("./data/easy_japanese2.csv")

# 使用するカラムを#日本語(原文),#やさしい日本語に絞る
df_1 = df_1[["#日本語(原文)", "#やさしい日本語"]]
df_2 = df_2[["#日本語(原文)", "#やさしい日本語"]]

# 2つのデータフレームを結合する
df = pd.concat([df_1, df_2])

# 結合したデータフレームをシャッフルする
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# カラムをinput, targetに変更する
df.columns = ["input", "target"]

# 結合したデータフレームをtrainとvalidに分割
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

# validとtestに分割
df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=42)

# train, valid, testのデータフレームを保存する
df_train.to_csv("./data/train.csv", index=False)
df_valid.to_csv("./data/valid.csv", index=False)
df_test.to_csv("./data/test.csv", index=False)