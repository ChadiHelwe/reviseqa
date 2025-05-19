import pandas as pd

csv_path = "/Users/sultan/Documents/Github/kaust/reviseqa/src/results/new/google_gemini-2.5-flash-preview_20250508_113359_token_count_stats.csv"
df = pd.read_csv(csv_path)
per_chain = df.groupby("chain_idx")["token_count"].sum()
print("Total tokens per chain_idx:")
print(per_chain)
mean_tokens = per_chain.mean()
print(f"\nMean tokens consumed per chain_idx: {mean_tokens:.2f}")
