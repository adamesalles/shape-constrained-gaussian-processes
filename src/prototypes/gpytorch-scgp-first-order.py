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
from torchinfo import summary

PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = PATH / "data"
LOGGING = False

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Set default device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cpu = torch.device("cpu")
# torch.set_default_device(device)
print(f'Using {device} as device')


# Loading data
def load_data(path: str) -> tuple:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    x_train = torch.from_numpy(data[:, 0])
    y_train = torch.stack(
        [torch.from_numpy(data[:, 1]), torch.from_numpy(data[:, 2])], -1
    ).squeeze(1)
    
    # verify if the data is convex on y
    x_train, indices = x_train.sort()
    y_train = y_train[indices]
    
    diff = y_train[1:, 1] - y_train[:-1, 1]
    assert not (torch.any(diff < 0) & torch.any(diff >= 0)), \
        "Data is not convex or concave"
    
    return x_train, y_train


# Read http://www.gaussianprocess.org/gpml/chapters/RW9.pdf
class SCGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(SCGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                  covar_x)


def scgp_fit(path: str, iters: int, kernel: gpytorch.kernels.Kernel) -> tuple:
    train_x, train_y = load_data(path)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2,
    )  # y and y_prime
    model = SCGP(train_x, train_y, likelihood, kernel)
    
    # sending to cuda
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Optimizing hyperparameters via marginal log likelihood
    model.train()
    likelihood.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in trange(iters):
        
        # print({n: p for n, p in model.named_parameters()})
        # if i == 1:
        #     for n, p in model.named_parameters():
        #         if n == 'likelihood.raw_task_noises':
        #             initial_grad = p.grad
        #             # turn off derivative grad
        #             new_grad = torch.zeros_like(initial_grad)
        #             new_grad[0] = initial_grad[0]
        #             tqdm.write(f"Initial grad: {new_grad}")
        #             p.grad = new_grad
        # elif i == iters // 3:
        #     for n, p in model.named_parameters():
        #         if n == 'likelihood.raw_task_noises':
        #             second_grad = p.grad
        #             newer_grad = torch.zeros_like(initial_grad)
        #             newer_grad[1] = initial_grad[1]
        #             p.grad = newer_grad
        # elif i == (2 * iters) // 3:
        #     for n, p in model.named_parameters():
        #         if n == 'likelihood.raw_task_noises':
        #             third_grad = p.grad
        #             final_grad = torch.zeros_like(initial_grad)
        #             final_grad[0] = second_grad[0]
        #             final_grad[1] = third_grad[1]
        #             p.grad = final_grad
               
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        noise = model.likelihood.noise.item()
        if LOGGING or i == iters - 1:
            tqdm.write(
                f"Iter {i + 1}/{iters} - Loss: {loss.item():.3f} "
                f"noise: {noise:.3f}"
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
    
    # Name of the plot
    name = f"SCGP_Scale{ker_name}_{dataset_name}_test"

    # Evaluation mode
    model = model.cpu()
    likelihood = likelihood.cpu()
    model.train()
    model.eval()
    likelihood.eval()

    # Initialize plots
    f, (y_ax, y_prime_ax) = plt.subplots(1, 2,
                                         figsize=(10, 4), tight_layout=True)

    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    test_y = test_y.cpu()
    
    
    # Make predictions
    mse = torch.nn.MSELoss()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        if plot_mse:
            mse_loss = mse(mean[:, 0], test_y[:, 0])
            mse_loss_prime = mse(mean[:, 1], test_y[:, 1])
    
    # Plotting predictions for f
    y_ax.plot(train_x.numpy(), train_y[:, 0].numpy(), "k*")
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
    if plot_mse:
        # show mse
        y_ax.text(0.05, 0.95, f"MSE: {mse_loss:.2f}",
                  transform=y_ax.transAxes)

    # Plotting predictions for f'
    y_prime_ax.plot(train_x.numpy(), train_y[:, 1].numpy(),
                    "k*")
    y_prime_ax.plot(test_x.numpy(), mean[:, 1].numpy(), "b")
    y_prime_ax.fill_between(
        test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5
    )
    y_prime_ax.legend(["Observed Derivatives", "Mean", "Confidence"])
    y_prime_ax.set_title(r"Derivatives with respect to $\alpha$")
    y_prime_ax.set_xlim([0, 1])
    
    y_prime_ax.set_xlabel(r"$\alpha$")
    y_prime_ax.set_ylabel(r"$\frac{\mathrm{d}}{\mathrm{d}\alpha}f_{\boldsymbol{z}}(\alpha)$")
    if plot_mse:
        # show mse
        y_prime_ax.text(0.05, 0.95, f"MSE: {mse_loss_prime:.2f}",
                        transform=y_prime_ax.transAxes)

    save_path = PATH / "results" /  str(dataset_name) / 'scgp' 
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
    
    kernels = {"RBFKernel": gpytorch.kernels.RBFKernelGrad}
    # for i in range(3, 7):
    #     kernels[f"PolynomialKernel{i}"] = \
    #         gpytorch.kernels.PolynomialKernelGrad(power=i)

    for ker_name, kernel in kernels.items():
        print(f"Running {ker_name}")

        for dataset_name, dataset_path in datasets.items():
            start_time = time.time()
            try:
                kernel_scgp = kernel()
                train_x, train_y, data_scgp, data_likelihood = scgp_fit(
                    dataset_path, 2000, kernel=kernel_scgp
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
                    f"Finished {ker_name} on {dataset_name} took {final_time:.2f}s"
                )
                del train_x, train_y, data_scgp, data_likelihood, kernel_scgp
            except Exception as e:
                print(f"Error on {ker_name} on {dataset_name} took {e}")
                continue

