import multiprocessing
from train import run_experiment1
import matplotlib.pyplot as plt
import numpy as np
from plot import plot_sam_comparison
from sam1 import SAM1 

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # 运行原始SAM实验
    print("Running original SAM experiment...")
    sam_metrics = run_experiment1("SAM", num_epochs=100)

    # 运行改进SAM1实验
    print("\nRunning improved SAM1 experiment...")
    sam1_metrics = run_experiment1("SAM1", num_epochs=100)

    # 绘制对比图
    plot_sam_comparison(sam_metrics, sam1_metrics, num_epochs=100)
    print("Comparison plot saved as 'sam_vs_sam1_comparison.png'")