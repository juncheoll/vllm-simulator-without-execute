import pandas as pd

file_path = 'squad_sorted-p1.csv'
num_top = 4096
num_bottom = 4096

df = pd.read_csv(file_path)

top_rows = df.head(num_top)
bottom_rows = df.tail(num_bottom)

rows = pd.concat([top_rows, bottom_rows])
rows = rows.sample(frac=1)

output_file = 'shuffled_squad1.csv'
rows.to_csv(output_file, index=False)