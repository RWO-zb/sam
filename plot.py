# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(opt_name, metrics):
    """绘制单个优化器的训练曲线"""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy', linestyle='--')
    plt.plot(epochs, metrics['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{opt_name} Accuracy')
    plt.xticks(np.arange(0, len(epochs)+1, 5))  # 只显示0和5的倍数
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss', linestyle='--')
    plt.plot(epochs, metrics['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt_name} Loss')
    plt.xticks(np.arange(0, len(epochs)+1, 5))  # 只显示0和5的倍数
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{opt_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(optimizers=["SGD", "Adam", "SAM"]):
    """绘制多个优化器的比较曲线"""
    plt.figure(figsize=(15, 10))
    epochs = None
    
    # 加载所有数据
    all_data = {}
    for opt in optimizers:
        try:
            data = np.loadtxt(f'{opt}_metrics.txt', delimiter=',', skiprows=1)
            all_data[opt] = {
                'train_loss': data[:, 0],
                'train_acc': data[:, 1],
                'test_loss': data[:, 2],
                'test_acc': data[:, 3]
            }
            if epochs is None:
                epochs = range(1, len(data[:, 0]) + 1)
        except:
            print(f"Warning: Could not load data for {opt}")
            continue
    
    if not all_data:
        return
    
    # 设置x轴刻度
    x_ticks = np.arange(0, len(epochs)+1, 5)  # 只显示0和5的倍数
    
    # 综合准确率比较
    plt.subplot(2, 2, 1)
    for opt in all_data.keys():
        plt.plot(epochs, all_data[opt]['train_acc'], label=f'{opt} Train', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy Comparison')
    plt.xticks(x_ticks)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for opt in all_data.keys():
        plt.plot(epochs, all_data[opt]['test_acc'], label=f'{opt} Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.xticks(x_ticks)
    plt.legend()
    
    # 综合损失比较
    plt.subplot(2, 2, 3)
    for opt in all_data.keys():
        plt.plot(epochs, all_data[opt]['train_loss'], label=f'{opt} Train', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Comparison')
    plt.xticks(x_ticks)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for opt in all_data.keys():
        plt.plot(epochs, all_data[opt]['test_loss'], label=f'{opt} Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.xticks(x_ticks)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(metrics, filename):
    """保存指标到文件"""
    data = np.column_stack((
        metrics['train_loss'], 
        metrics['train_acc'], 
        metrics['test_loss'], 
        metrics['test_acc']
    ))
    np.savetxt(filename, data, delimiter=',', 
               header='train_loss,train_acc,test_loss,test_acc', comments='')

def load_metrics(filename):
    """从文件加载指标"""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return {
        'train_loss': data[:, 0],
        'train_acc': data[:, 1],
        'test_loss': data[:, 2],
        'test_acc': data[:, 3]
    }


def plot_metrics_comparison(opt_name, all_metrics):
    """绘制同一种优化算法的不同参数配置比较图"""
    plt.figure(figsize=(15, 10))
    epochs = range(1, len(next(iter(all_metrics.values()))['train_loss']) + 1)
    
    # 设置颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    linestyles = ['-', '--', '-.', ':']
    
    # 训练准确率比较
    plt.subplot(2, 2, 1)
    for i, (config_name, metrics) in enumerate(all_metrics.items()):
        plt.plot(epochs, metrics['train_acc'], 
                label=f"{config_name}", 
                color=colors[i], 
                linestyle=linestyles[i % len(linestyles)])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{opt_name} Train Accuracy Comparison')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 测试准确率比较
    plt.subplot(2, 2, 2)
    for i, (config_name, metrics) in enumerate(all_metrics.items()):
        plt.plot(epochs, metrics['test_acc'], 
                label=f"{config_name}", 
                color=colors[i], 
                linestyle=linestyles[i % len(linestyles)])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{opt_name} Test Accuracy Comparison')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 训练损失比较
    plt.subplot(2, 2, 3)
    for i, (config_name, metrics) in enumerate(all_metrics.items()):
        plt.plot(epochs, metrics['train_loss'], 
                label=f"{config_name}", 
                color=colors[i], 
                linestyle=linestyles[i % len(linestyles)])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt_name} Train Loss Comparison')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 测试损失比较
    plt.subplot(2, 2, 4)
    for i, (config_name, metrics) in enumerate(all_metrics.items()):
        plt.plot(epochs, metrics['test_loss'], 
                label=f"{config_name}", 
                color=colors[i], 
                linestyle=linestyles[i % len(linestyles)])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt_name} Test Loss Comparison')
    plt.xticks(np.arange(0, len(epochs)+1, 5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{opt_name}_params_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()