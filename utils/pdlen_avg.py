import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

folder_path = "../datasets"
lengths = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            lengths.append(len(df))
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if lengths:
    arr = np.array(lengths)
    print(f"📊 統計資訊（共 {len(arr)} 個檔案）")
    print(f"- 平均值 (mean): {np.mean(arr):.2f}")
    print(f"- 中位數 (median): {np.median(arr):.2f}")
    print(f"- 眾數 (mode): {stats.mode(arr, keepdims=False).mode}（出現次數最多的長度）")
    print(f"- 標準差 (std): {np.std(arr):.2f}")
    print(f"- 最小值: {np.min(arr)}, 最大值: {np.max(arr)}")
    print(f"- Q1: {np.percentile(arr, 25)}, Q3: {np.percentile(arr, 75)}")

    # 畫出分布圖
    sns.histplot(arr, bins=30, kde=True)
    plt.title("CSV 檔案行數分布")
    plt.xlabel("每個 CSV 檔案的資料筆數")
    plt.ylabel("檔案數量")
    plt.grid(True)
    plt.show()
else:
    print("❌ 沒有成功讀取任何 CSV 檔案")
