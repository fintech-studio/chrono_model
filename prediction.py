from warnings import catch_warnings

import pandas as pd
import torch
from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.ipc as ipc


def gen_quantities_list(quantiles, dim=1):
    # 返回中位數列表
    quan_list = torch.median(quantiles.squeeze(), dim=dim).values.tolist()
    return quan_list

def gen_mean_list(mean):
    return mean[0].tolist()

def cal_mape(sample: list, answer: list):
    ape = [abs((a - s) / a) if a != 0 else 0 for s, a in zip(sample, answer)]
    mape = 100 * sum(ape) / len(sample)
    return mape

def cal_mse(sample: list, answer: list):
    squared_errors = [(s - a) ** 2 for s, a in zip(sample, answer)]
    mse = sum(squared_errors) / len(sample)
    return mse


def evaluate(page, context_length, prediction_length, model_path, evaluate_path):
    DataNotEnough = False
    # 加載預訓練的 pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    # 讀取 Arrow 文件
    file_path = evaluate_path

    with open(file_path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    # 轉換為 pandas dataframe
    df = table.to_pandas()

    lens = len(df["target"][page])
    if lens < 500:
        DataNotEnough = True

    forecast_index = []
    median = []

    index = context_length

    true_values = []
    pred_values = []

    while index < lens - prediction_length:
        context_data = df["target"][page][index - context_length: index].tolist()
        context_tensor = torch.tensor(context_data, dtype=torch.float32)

        # 預測分位數
        quantiles, mean = pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # 提取分位數（低、中、高分位）
        q_lower = quantiles[0, 0, 0].item()
        q_median = quantiles[0, 0, 1].item()
        q_upper = quantiles[0, 0, 2].item()

        forecast_index.append(index)

        true_val = df['target'][page][index:index + prediction_length]
        print(f"index={index} 預測中位數={q_median:.2f}，正確答案={true_val}")

        true_values.extend(true_val)
        # pred_values.extend(gen_quantities_list(quantiles))
        pred_values.extend(gen_mean_list(mean))

        index += prediction_length

    # 計算 MSE
    MSE = cal_mse(pred_values, true_values)
    MAPE = cal_mape(pred_values, true_values)

    # 計算 R²
    # true_values = np.array(true_values).flatten()
    # pred_values = np.array(pred_values).flatten()
    # mean_true = true_values.mean()
    # ss_total = np.sum((true_values - mean_true) ** 2)
    # ss_residual = np.sum((true_values - pred_values) ** 2)
    # R2 = 1 - (ss_residual / ss_total)

    # 作圖
    forecast_index = [i for i in range(context_length +1, len(pred_values) + context_length +1)]
    plt.figure(figsize=(10, 5))
    plt.plot(df["target"][page], label="Historical target", color="royalblue")
    plt.plot(forecast_index, pred_values, label="Median Forecast", color="tomato")

    plt.text(
        x=0,
        y=max(df["target"][page]) * 0.95,
        s=f"MSE: {MSE:.3f}\nMAPE: {MAPE:.3f}%",
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.xlabel("Index")
    plt.ylabel("Target Price")
    plt.legend()
    plt.grid()
    plt.show()

    # return {"outcome": pred_values,"DataNotEnough": DataNotEnough}

if __name__ == "__main__":
    try:
        evaluate(1, 192, 12, "output/gooood/checkpoint-final", "arrows/valid.arrow")
    except Exception as e:
        print(f"Error: {e}")