import pathlib
import time
import torch
import gpytorch
import pandas as pd
from scgp.utils import save_plot_scgp, scgp_fit, save_plot_gp, gp_fit

PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = PATH / "data"
LOGGING = False

if __name__ == "__main__":
    datasets = {
        "uniform_new_HC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_HC.csv",
        "uniform_new_MC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_MC.csv",
        "adaptive_new_HC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_HC.csv",
        "adaptive_new_MC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_MC.csv",
    }

    models = {
        "SCGP": {
            "kernel": gpytorch.kernels.RBFKernelGrad,
            "plot": save_plot_scgp,
            "fit": scgp_fit,
            "iters": 2000,
        },
        "GP": {
            "kernel": gpytorch.kernels.RBFKernel,
            "plot": save_plot_gp,
            "fit": gp_fit,
            "iters": 1000,
        },
    }
    
    scenarios = {
        "nobs": list(range(6, 21, 2)),
        # "nobs": [10, 20],
    }
    
    mse_df = pd.DataFrame(columns=["model", "scenario", "dataset", "mse"])
        
    for model_name, model in models.items():
        for scenario in scenarios["nobs"]:
            for dataset_name, dataset_path in datasets.items():
                start_time = time.time()
                try:
                    kernel = model["kernel"]()
                    train_x, train_y, data, data_likelihood = model["fit"](
                        dataset_path, model["iters"], kernel=kernel,
                        nobs=scenario
                    )
                    name = f"{model_name}_{scenario}_nobs"
                    mse = model["plot"](
                        train_x,
                        train_y,
                        data,
                        data_likelihood,
                        name,
                        dataset_name,
                    )
                    final_time = time.time() - start_time
                    
                    mse_df = pd.concat(
                        [
                            mse_df,
                            pd.DataFrame(
                                {
                                    "model": [model_name],
                                    "scenario": [scenario],
                                    "dataset": [dataset_name],
                                    "mse": [mse],
                                }
                            ),
                        ]
                    )
                    print(
                        f"Finished {model_name} on {dataset_name}"
                        f" with nobs={scenario}"
                        f", it took {final_time:.2f}s"
                    )
                    del train_x, train_y, data, data_likelihood, kernel, mse
                    
                except Exception as e:
                    print(f"Error on {model_name} on {dataset_name} | {e}")
                    continue
                
    # save mse_df
    # print("Saving mse_df")
    # mse_df.to_csv(PATH / "results" / "mse_df.csv", index=False)
    # print(mse_df)
