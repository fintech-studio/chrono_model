import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

# 替換成你的檔案路徑
file_path = "../arrows/valid.arrow"

# 讀取 Arrow 檔
with open(file_path, "rb") as f:
    reader = ipc.RecordBatchFileReader(f)
    table = reader.read_all()

# 轉成 pandas dataframe
df = table.to_pandas()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# 顯示前幾筆
# print(df.head())
# lens = len(df["target"][0])
# print(lens)
print(df.iloc[856])