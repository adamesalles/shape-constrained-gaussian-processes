# Inspired by https://docs.gpytorch.ai/en/stable/examples/
# 08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html

import torch
import gpytorch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import time
from tqdm import tqdm, trange

PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = PATH / "data"
LOGGING = True

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# Set default device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f'Using {device} as device')


# Loading data
def load_data(path: str) -> tuple:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    x_train = torch.from_numpy(data[:, 0]).to(device)
    y_train = torch.from_numpy(data[:, 1]).to(device)
    return x_train, y_train


# Read http://www.gaussianprocess.org/gpml/chapters/RW9.pdf
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gp_fit(path: str, iters: int, kernel: gpytorch.kernels.Kernel) -> tuple:
    train_x, train_y = load_data(path)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()  # y 
    model = GP(train_x, train_y, likelihood, kernel)

    # Optimizing hyperparameters via marginal log likelihood
    model.train()
    likelihood.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in trange(iters):
        # print({n: p.grad for n, p in model.named_parameters()})
        
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        noise = model.likelihood.noise.item()
        if LOGGING:
            tqdm.write(
                f"Iter {i + 1}/{iters} - Loss: {loss.item():.3f} "
                f"noise: {noise:.3f}"
            )
        optimizer.step()

    return train_x, train_y, model, likelihood


def save_plot_scpg(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: GP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    ker_name: str,
    dataset_name: str
) -> None:

    # Name of the plot
    name = f"GP_Scale{ker_name}_{dataset_name}_test"

    global true_HC_x, true_MC_x, true_HC_y, true_MC_y
    
    
    plot_mse = False
    if dataset_name == "uniform_new_HC" or dataset_name == "adaptive_new_HC":
        test_x = true_HC_x
        test_y = true_HC_y
        plot_mse = True
    elif dataset_name == "uniform_new_MC" or dataset_name == "adaptive_new_MC":
        test_x = true_MC_x
        test_y = true_MC_y
        plot_mse = True
    else:
        test_x = torch.linspace(0, 1, 100)
    
    # Evaluation mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    f, y_ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)

    # Make predictions
    mse = torch.nn.MSELoss()
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(100):
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        if plot_mse:
            mse_loss = mse(mean, test_y)

    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    mean = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()
    
    # Plotting predictions for f
    y_ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    y_ax.plot(test_x.numpy(), mean.numpy(), "b")
    y_ax.fill_between(
        test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5
    )
    y_ax.legend(["Observed Values", "Mean", "Confidence"])
    y_ax.set_title("Function values")
    y_ax.set_xlim([0, 1])
    # y_ax.set_ylim([-7.5, 12.5])
    y_ax.set_xlabel(r"$\alpha$")
    y_ax.set_ylabel(r"$f_{\boldsymbol{z}}(\alpha)$")
    if plot_mse:
        # show mse
        y_ax.text(0.05, 0.95, f"MSE: {mse_loss:.2f}",
                  transform=y_ax.transAxes)

    save_path = PATH / "results" / str(dataset_name) / 'gp'
    save_path.mkdir(parents=True, exist_ok=True)

    f.savefig(save_path / (name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    datasets = {
        # "uniform": DATA_PATH / "Gaussian_logCA0_uniform_J=20.csv",
        # "adaptive": DATA_PATH / "Gaussian_logCA0_adaptive_J=20.csv",
        # "uniform_HC": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20.csv",
        # "adaptive_HC": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20.csv",
        # "uniform_HC2": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20_HC2.csv",
        # "adaptive_HC2": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20_HC2.csv",
        # "uniform_HC3": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20_HC3.csv",
        # "adaptive_HC3": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20_HC3.csv",
        "uniform_new_HC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_HC.csv",
        # "uniform_new_MC": DATA_PATH / "Gaussian_logCA0_uniform_J=20_MC.csv",
        # "adaptive_new_HC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_HC.csv",
        # "adaptive_new_MC": DATA_PATH / "Gaussian_logCA0_adaptive_J=20_MC.csv",
    }
    
    true_HC_x, true_HC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_HC.csv")
    true_MC_x, true_MC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_MC.csv")

    kernels = {"RBFKernel": gpytorch.kernels.RBFKernel}
    # for i in range(3, 7):
    #     kernels[f"PolynomialKernel{i}"] = \
    #         gpytorch.kernels.PolynomialKernel(power=i)

    for ker_name, kernel in kernels.items():
        print(f"Running {ker_name}")

        for dataset_name, dataset_path in datasets.items():
            start_time = time.time()
            try:
                kernel_gp = kernel()
                train_x, train_y, data_gp, data_likelihood = gp_fit(
                    dataset_path, 1000, kernel=kernel_gp
                )
                save_plot_scpg(
                    train_x,
                    train_y,
                    data_gp,
                    data_likelihood,
                    ker_name,
                    dataset_name
                )
                final_time = time.time() - start_time
                print(
                    f"Finished {ker_name} on {dataset_name} took {final_time:.2f}s"
                )
                
                del train_x, train_y, data_gp, data_likelihood, kernel_gp
            except Exception as e:
                print(f"Error on {ker_name} on {dataset_name} took {e}")
                continue
