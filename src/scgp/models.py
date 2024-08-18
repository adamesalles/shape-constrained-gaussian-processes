import torch
import gpytorch
from tqdm import tqdm, trange


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, iters: int, kernel: gpytorch.kernels.Kernel, log: bool):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Optimizing hyperparameters via marginal log likelihood
        self.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
        
        for i in trange(iters):
            optimizer.zero_grad()
            output = self.forward(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if log:
                tqdm.write(
                    f"Iter {i + 1}/{iters} - Loss: {loss.item():.3f}"
                    "noise: {model.likelihood.noise.item():.3f}"
                )
            optimizer.step()

        return self.train_x, self.train_y, self, likelihood
    

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