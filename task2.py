import multiprocessing
from train import run_experiment2
from plot import plot_metrics_comparison, plot_comparison

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    optimizers = {
        "SGD": [
            {'lr': 0.1, 'momentum': 0.9},
            {'lr': 0.05, 'momentum': 0.9},
            {'lr': 0.1, 'momentum': 0.8}
        ],
        "Adam": [
            {'lr': 3e-4},
            {'lr': 1e-4},
            {'lr': 5e-4}
        ],
        "SAM": [
            {'rho': 0.1, 'lr': 0.05, 'momentum': 0.9},
            {'rho': 0.05, 'lr': 0.05, 'momentum': 0.9},
            {'rho': 0.2, 'lr': 0.05, 'momentum': 0.9}
        ]
    }
    
    all_results = {}
    
    for opt_name, params_list in optimizers.items():
        metrics = run_experiment2(opt_name, num_epochs=100, params_list=params_list)
        all_results[opt_name] = metrics
        plot_metrics_comparison(opt_name, metrics)
    
    # 绘制不同优化器之间的比较（使用默认参数配置）
    default_metrics = {opt: metrics[f"{opt}_config_1"] for opt, metrics in all_results.items()}
    plot_comparison(default_metrics.keys())