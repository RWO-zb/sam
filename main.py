# main.py
import multiprocessing
from train import run_experiment
from plot import plot_metrics, plot_comparison

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    optimizers = ["SGD", "Adam", "SAM"]
    all_metrics = {}
    
    for opt in optimizers:
        metrics = run_experiment(opt, num_epochs=10)
        all_metrics[opt] = metrics
        plot_metrics(opt, metrics)
    
    plot_comparison(optimizers)