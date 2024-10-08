# Inspired by https://docs.gpytorch.ai/en/stable/examples/
# 08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html

import torch
import gpytorch
import matplotlib
import numpy as np
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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
# print(f'Using {device} as device')


# Loading data
def load_data(path: str) -> tuple:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    x_train = torch.from_numpy(data[:, 0])
    y_train = torch.stack(
        [torch.from_numpy(data[:, 1]),
         torch.from_numpy(data[:, 2]),
         torch.from_numpy(data[:, 3])], -1
    ).squeeze(1)
    
    # verify if the data is convex on y by the signal of the second derivative
    assert (torch.all(y_train[:, 2] >= 0)
            or torch.all(y_train[:, 2] <= 0)), \
        "Data is not convex or concave"
    
    return x_train, y_train


# Read http://www.gaussianprocess.org/gpml/chapters/RW9.pdf
class SCGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(SCGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGradGrad()
        self.base_kernel = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # https://docs.gpytorch.ai/en/latest/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                  covar_x)


def scgp_fit(path: str, iters: int, kernel) -> tuple:
    train_x, train_y = load_data(path)

    # y, y_prime and y_double_prime
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = SCGP(train_x, train_y, likelihood, kernel)

    # Optimizing hyperparameters via marginal log likelihood
    model.train()
    likelihood.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in trange(iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        if LOGGING:
            tqdm.write(
                f"Iter {i + 1}/{iters} - Loss: {loss.item():.3f}"
                f" lengthscale: {lengthscale:.3f}"
                f" noise: {model.likelihood.noise.item():.3f}"
            )
        optimizer.step()

    return train_x, train_y, model, likelihood


def save_plot_scpg(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: SCGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    ker_name: str,
    dataset_name: str
) -> None:

    # Name of the plot
    name = f"SCGP_SO_Scale{ker_name}_{dataset_name}_test"

    # Evaluation mode
    model.train()
    model.eval()
    likelihood.eval()

    # Initialize plots
    f, (y_ax, y_prime_ax, y_double_prime_ax) = plt.subplots(1, 3,
                                                            figsize=(15, 4))

    # Make predictions
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(100):
        test_x = torch.linspace(0, 1, 100)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # Plotting predictions for f
    y_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), "k*")
    y_ax.plot(test_x.numpy(), mean[:, 0].numpy(), "b")
    y_ax.fill_between(
        test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5
    )
    y_ax.legend(["Observed Values", "Mean", "Confidence"])
    y_ax.set_title("Function values")
    y_ax.set_xlim([0, 1])
    # y_ax.set_ylim([-7.5, 12.5])
    y_ax.set_xlabel(r"$\alpha$")
    y_ax.set_ylabel(r"$f_{\boldsymbol{z}}(\alpha)$")

    # Plotting predictions for f'
    y_prime_ax.plot(train_x.detach().numpy(),
                    train_y[:, 1].detach().numpy(), "k*")
    y_prime_ax.plot(test_x.numpy(), mean[:, 1].numpy(), "b")
    y_prime_ax.fill_between(
        test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5
    )
    y_prime_ax.legend(["Observed First Order Derivatives",
                       "Mean", "Confidence"])
    y_prime_ax.set_title(r"Derivatives with respect to $\alpha$")
    y_prime_ax.set_xlim([0, 1])
    
    y_prime_ax.set_xlabel(r"$\alpha$")
    y_prime_ax.set_ylabel(r"$\frac{\mathrm{d}}{\mathrm{d}\alpha}f_{\boldsymbol{z}}(\alpha)$")

    # Plotting predictions for f'
    y_double_prime_ax.plot(
        train_x.detach().numpy(), train_y[:, 2].detach().numpy(), "k*"
    )
    y_double_prime_ax.plot(test_x.numpy(), mean[:, 2].numpy(), "b")
    y_double_prime_ax.fill_between(
        test_x.numpy(), lower[:, 2].numpy(), upper[:, 2].numpy(), alpha=0.5
    )
    y_double_prime_ax.legend(
        ["Observed Second Order Derivatives", "Mean", "Confidence"]
    )
    y_double_prime_ax.set_title(r"Second Order Derivatives with respect to $\alpha$")
    y_double_prime_ax.set_xlim([0, 1])
    
    y_double_prime_ax.set_xlabel(r"$\alpha$")
    y_double_prime_ax.set_ylabel(r"$\frac{\mathrm{d}^2}{\mathrm{d}\alpha^2}f_{\boldsymbol{z}}(\alpha)$")
    
    save_path = PATH / "results" / str(dataset_name) / 'scgp_so'
    save_path.mkdir(parents=True, exist_ok=True)

    f.savefig(save_path / (name + ".png"))
    plt.close(f)


if __name__ == "__main__":
    datasets = {
        "uniform": DATA_PATH / "Gaussian_logCA0_uniform_J=20.csv",
        "adaptive": DATA_PATH / "Gaussian_logCA0_adaptive_J=20.csv",
        "uniform_HC": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20.csv",
        "adaptive_HC": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20.csv",
        "uniform_HC2": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20_HC2.csv",
        "adaptive_HC2": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20_HC2.csv",
        "uniform_HC3": DATA_PATH / "Gaussian_HC_logCA0_uniform_J=20_HC3.csv",
        "adaptive_HC3": DATA_PATH / "Gaussian_HC_logCA0_adaptive_J=20_HC3.csv",
    }

    kernels = {"RBFKernel": gpytorch.kernels.RBFKernelGradGrad()}

    for ker_name, kernel in kernels.items():
        print(f"Running {ker_name}")
        for dataset_name, dataset_path in datasets.items():
            start_time = time.time()
            train_x, train_y, data_scgp, data_likelihood = scgp_fit(
                dataset_path, 100, kernel=kernel
            )
            save_plot_scpg(
                train_x,
                train_y,
                data_scgp,
                data_likelihood,
                ker_name,
                dataset_name
            )
            final_time = time.time() - start_time
            print(
                f"Finished {ker_name} for {dataset_name} in {final_time:.2f} s"
            )
