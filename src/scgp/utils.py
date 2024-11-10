import torch
import gpytorch
import matplotlib
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scgp.models import GP, SCGP

PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = PATH / "data"
LOGGING = False
COLORS = ['#377eb8', '#ff7f00']

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
    data = np.loadtxt(path, delimiter=",", skiprows=2, dtype=np.float32)
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


def remove_points(train_x, train_y, nobs):
    """
    Remove points from the dataset equally spaced
    """
    idx = torch.linspace(0, len(train_x) - 1, nobs).int()
    return train_x[idx], train_y[idx]


def scgp_fit(path: str, iters: int, kernel: gpytorch.kernels.Kernel,
             nobs: int = 0) -> tuple:
    train_x, train_y = load_data(path)
    
    if nobs:
        train_x, train_y = remove_points(train_x, train_y, nobs)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2
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


# Load true data
true_HC_x, true_HC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_HC.csv")
true_MC_x, true_MC_y = load_data(DATA_PATH / "true_Gaussian_logCA0_MC.csv")


def save_plot_scgp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: SCGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    name: str,
    dataset_name: str,
    save_path: pathlib.Path | None = None
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
    y_ax.plot(test_x.numpy(), mean[:, 0].numpy(), color=COLORS[1])
    y_ax.fill_between(
        test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5,
        color=COLORS[1]
    )
    y_ax.plot(train_x.numpy(), train_y[:, 0].numpy(), "k*")

    y_ax.legend(["Observed Values", "Mean", "Confidence"],
                loc="upper left")
    y_ax.set_title("Function values")
    y_ax.set_xlim([0, 1])
    # y_ax.set_ylim([-7.5, 12.5])
    y_ax.set_xlabel(r"$\alpha$")
    y_ax.set_ylabel(r"$f_{\boldsymbol{z}}(\alpha)$")
    if plot_mse:
        # show mse in bottom right (larger font)
        y_ax.text(0.5, 0.05, f"MSE: {mse_loss:.2f}",
                  transform=y_ax.transAxes,
                    fontsize=16,
                    bbox=dict(facecolor=COLORS[1], alpha=0.5))

    # Plotting predictions for f'
    y_prime_ax.plot(test_x.numpy(), mean[:, 1].numpy(), color=COLORS[1])
    y_prime_ax.fill_between(
        test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5,
        color=COLORS[1]
    )
    y_prime_ax.plot(train_x.numpy(), train_y[:, 1].numpy(),
                    "k*")

    y_prime_ax.legend(["Observed Derivatives", "Mean", "Confidence"],
                      loc="upper left")
    y_prime_ax.set_title(r"Derivatives with respect to $\alpha$")
    y_prime_ax.set_xlim([0, 1])
    
    y_prime_ax.set_xlabel(r"$\alpha$")
    y_prime_ax.set_ylabel(r"$\frac{\mathrm{d}}{\mathrm{d}\alpha}f_{\boldsymbol{z}}(\alpha)$")
    if plot_mse:
        # show mse
        y_prime_ax.text(0.5, 0.05, f"MSE: {mse_loss_prime:.2f}",
                  transform=y_prime_ax.transAxes,
                    fontsize=16,
                    bbox=dict(facecolor=COLORS[1], alpha=0.5))

    save_path = PATH / "experiments" / str(dataset_name) 
    save_path.mkdir(parents=True, exist_ok=True)

    f.suptitle("Shape-constrained Gaussian Process (SCGP)",
               fontsize=18,
               color=COLORS[1],
               fontweight='bold')
    
    # f.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)

    # f.patch.set_linewidth(6)
    # f.patch.set_edgecolor(COLORS[1])
    
    # save csv with test data
    if save_path:
        pd.DataFrame(
            {
                "x": test_x.numpy(),
                "f": test_y[:, 0].numpy(),
                "f_prime": test_y[:, 1].numpy(),
                "mean": mean[:, 0].numpy(),
                "mean_prime": mean[:, 1].numpy(),
                "lower": lower[:, 0].numpy(),
                "lower_prime": lower[:, 1].numpy(),
                "upper": upper[:, 0].numpy(),
                "upper_prime": upper[:, 1].numpy(),
            }
        ).to_csv(save_path / (name + "_.csv"), index=False)
    f.savefig(save_path / (name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(f)
    
    if plot_mse:
        return mse_loss.item()

    
def gp_fit(path: str, iters: int, kernel: gpytorch.kernels.Kernel,
           nobs: int = 0) -> tuple:
    train_x, train_y = load_data(path)
    train_y = train_y[:, 0]
    if nobs:
        train_x, train_y = remove_points(train_x, train_y, nobs)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()  # y 
    model = GP(train_x, train_y, likelihood, kernel)
    
    # Optimizing hyperparameters via marginal log likelihood
    model.train()
    likelihood.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  weight_decay=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in trange(iters):
        # print([param.cpu() for param in model.parameters()])
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


def save_plot_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model: GP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    name: str,
    dataset_name: str
) -> None:

    global true_HC_x, true_MC_x, true_HC_y, true_MC_y
    
    plot_mse = False
    if dataset_name == "uniform_new_HC" or dataset_name == "adaptive_new_HC":
        test_x = true_HC_x
        test_y = true_HC_y[:, 0]
        plot_mse = True
    elif dataset_name == "uniform_new_MC" or dataset_name == "adaptive_new_MC":
        test_x = true_MC_x
        test_y = true_MC_y[:, 0]
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
    y_ax.plot(test_x.numpy(), mean.numpy(), color=COLORS[0])
    y_ax.fill_between(
        test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5,
        color=COLORS[0]
    )
    y_ax.plot(train_x.numpy(), train_y.numpy(), "k*")
    y_ax.legend(["Observed Values", "Mean", "Confidence"])
    y_ax.set_title("Function values")
    y_ax.set_xlim([0, 1])
    # y_ax.set_ylim([-7.5, 12.5])
    y_ax.set_xlabel(r"$\alpha$")
    y_ax.set_ylabel(r"$f_{\boldsymbol{z}}(\alpha)$")
    if plot_mse:
        # show mse
        y_ax.text(0.5, 0.05, f"MSE: {mse_loss:.2f}",
                  transform=y_ax.transAxes,
                  fontsize=16,
                  bbox=dict(facecolor=COLORS[0], alpha=0.5))
    save_path = PATH / "experiments" / str(dataset_name)
    save_path.mkdir(parents=True, exist_ok=True)

    f.suptitle("Gaussian Process (GP)",
               fontsize=18,
               color=COLORS[0],
               fontweight='bold')
    
    # f.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)

    # f.patch.set_linewidth(6)
    # f.patch.set_edgecolor(COLORS[0])
    if save_path:
        pd.DataFrame(
            {
                "x": test_x.numpy(),
                "f": test_y.numpy(),
                "mean": mean.numpy(),
                "lower": lower.numpy(),
                "upper": upper.numpy(),
            }
        ).to_csv(save_path / (name + "_.csv"), index=False)
    f.savefig(save_path / (name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(f)
    
    if plot_mse:
        return mse_loss.item()