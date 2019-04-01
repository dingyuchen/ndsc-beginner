import pandas as pd

fashion_df = pd.read_csv("./train.csv")
is_fashion = fashion_df["Category"] > 16 or fashion_df['Category'] < 31
print(fashion_df[is_fashion])