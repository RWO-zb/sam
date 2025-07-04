import multiprocessing
from train import run_experiment1
from plot import plot_metrics
import matplotlib.pyplot as plt
import numpy as np

def plot_sam_comparison(sam_metrics, sam1_metrics, num_epochs):
    """绘制SAM和SAM1的对比曲线"""
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, sam_metrics['train_acc'], label='SAM Train', linestyle='--', color='blue')
    plt.plot(epochs, sam_metrics['test_acc'], label='SAM Test', color='blue')
    plt.plot(epochs, sam1_metrics['train_acc'], label='SAM1 Train', linestyle='--', color='red')
    plt.plot(epochs, sam1_metrics['test_acc'], label='SAM1 Test', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('SAM vs SAM1 Accuracy')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, sam_metrics['train_loss'], label='SAM Train', linestyle='--', color='blue')
    plt.plot(epochs, sam_metrics['test_loss'], label='SAM Test', color='blue')
    plt.plot(epochs, sam1_metrics['train_loss'], label='SAM1 Train', linestyle='--', color='red')
    plt.plot(epochs, sam1_metrics['test_loss'], label='SAM1 Test', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SAM vs SAM1 Loss')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sam_vs_sam1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # 运行原始SAM实验
    print("Running original SAM experiment...")
    sam_metrics = run_experiment1("SAM", num_epochs=100)
    
    # 运行改进SAM1实验
    print("\nRunning improved SAM1 experiment...")
    # 临时替换SAM为SAM1进行实验
    from sam1 import SAM1 as SAM1  # 现在SAM指向SAM1
    
    sam1_metrics = run_experiment1("SAM1", num_epochs=100)
    
    # 恢复原始SAM
    
    # 绘制对比图
    plot_sam_comparison(sam_metrics, sam1_metrics, num_epochs=100)
    
    print("Comparison plot saved as 'sam_vs_sam1_comparison.png'")