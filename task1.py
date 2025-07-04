# main.py
import multiprocessing
from train import run_experiment1
from plot import plot_metrics, plot_comparison

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    optimizers = ["SAM","SGD", "Adam" ]
    all_metrics = {}
    
    for opt in optimizers:
        metrics = run_experiment1(opt, num_epochs=5)
        all_metrics[opt] = metrics
        plot_metrics(opt, metrics)
    
    plot_comparison(optimizers)