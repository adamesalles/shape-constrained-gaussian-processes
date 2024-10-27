import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pathlib

# Define colors
COLORS = ['#4daf4a']

# Paths
PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = PATH / "data"
LOGGING = False

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

datasets = {
        "uniform_new_HC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_HC.csv",
        "uniform_new_MC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_MC.csv",
        "adaptive_new_HC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_HC.csv",
        "adaptive_new_MC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_MC.csv",
    }

# Loading data
def load_data(path: str) -> tuple:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    x_train = data[:, 0]
    y_train = data[:, 1]
    
    return x_train, y_train

def remove_points(train_x, train_y, nobs):
    """
    Remove points from the dataset equally spaced
    """
    idx = np.linspace(0, len(train_x) - 1, nobs).astype(int)
    return train_x[idx], train_y[idx]


true_MC_x, true_MC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_MC.csv")
true_HC_x, true_HC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_HC.csv")

model = 'SCAM'
if __name__ == '__main__':
    for dataset_name, dataset_path in datasets.items():
        if 'HC' in dataset_name:
            x_true, y_true = true_HC_x, true_HC_y
        else:
            x_true, y_true = true_MC_x[1:], true_MC_y[1:]
            
        for nobs in range(10, 21, 2) + [21]:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
            name = f"{model}_{nobs}_nobs"
            io_path = PATH / "experiments" / str(dataset_name)
            io_path.mkdir(parents=True, exist_ok=True)
            
            train_x, train_y = load_data(dataset_path)
            train_x, train_y = remove_points(train_x, train_y, nobs)
            
            test_x, test_y = load_data(io_path / f'{name}_.csv')
            mse_loss = np.mean((test_y - y_true)**2)
            
            ax.plot(x_true, y_true, color=COLORS[0])
            ax.plot(train_x, train_y, "k*")
            ax.legend(["Observed Values", "Mean", "Confidence"])
            ax.set_title("Function values")
            ax.set_xlim([0, 1])
            # ax.set_ylim([-7.5, 12.5])
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$f_{\boldsymbol{z}}(\alpha)$")
            ax.text(0.5, 0.05, f"MSE: {mse_loss:.2f}",
                    transform=ax.transAxes,
                    fontsize=16,
                    bbox=dict(facecolor=COLORS[0], alpha=0.5))
            
            fig.suptitle("Shape Constrained Additive Model (SCAM)",
               fontsize=18,
               color=COLORS[0],
               fontweight='bold')

            # plt.show()
            fig.savefig(io_path / (name + ".png"), dpi=600, bbox_inches="tight")
            plt.close(fig)