import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import click
import pdb
import pickle
import os

from hyperopt import hp, fmin, tpe, space_eval, Trials


class QDistribution:
    def __init__(self, theta, sigma=1):
        self.theta = theta
        self.sigma = sigma

    def Pr_x1_less_x2(self, delta):
        """
        P(x1 < x2) = exp(f(x2)) / (exp(f(x1)) + exp(f(x2)))
        delta = x1 - x2
        """
        return 1 / (1 + np.exp(np.dot(self.theta, delta.T) / self.sigma))


def log_sigmoid(x):
    """
    Numerically stable log sigmoid function.
    Computes log(sigmoid(x)) in a stable manner to avoid overflow.
    """
    result = np.zeros_like(x)

    idx = x >= 0
    result[idx] = -np.log(1 + np.exp(-x[idx]))

    idx = x < 0
    result[idx] = x[idx] - np.log(1 + np.exp(x[idx]))

    return result


def loss_MLE(theta, sigma, Delta_table_rpt, Samples):
    loss = -log_sigmoid(-Samples * np.dot(Delta_table_rpt, theta) / sigma)
    return loss.mean()


def loss_MLE_reg(theta, sigma, Delta_table_rpt, Samples, beta):
    loss = loss_MLE(theta, sigma, Delta_table_rpt, Samples)
    loss += beta * np.linalg.norm(theta, ord=1)
    return loss


def fista(gradient, prox, x0, L, max_iter=1000, tol=1e-6):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    Parameters:
    - gradient: Gradient function of the smooth function f.
    - prox: Proximal operator of the non-smooth function g.
    - x0: Initial guess for x.
    - L: Lipschitz constant of the gradient of f.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - x: Solution of the optimization problem.
    """

    x = x0.copy()
    y = x0.copy()
    t = 1.0

    for i in range(max_iter):
        x_prev = x.copy()

        grad_y = gradient(y)
        x = y - (1 / L) * grad_y

        x = prox(x, L)

        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_prev)

        if np.linalg.norm(x - x_prev) < tol:
            break

        t = t_next

    return x


class Problem:
    def __init__(self, m, k, d, sigma, beta, seed) -> None:
        self.d = d
        self.k = k
        self.m = m
        self.sigma = sigma
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.theta = self._generate_sparse_theta_star(self.rng, k=k, d=d)
        self.theta_init = self.rng.standard_normal(d)

        self.Q = QDistribution(self.theta, sigma=sigma)
        self.Delta_table, self.samples = self._generate_samples(
            self.Q, self.rng, d=d, m=m
        )
        def _unit_ball(theta):
            return 1 - np.linalg.norm(theta)

        self.constraints = {"type": "ineq", "fun": _unit_ball}

    def _generate_sparse_theta_star(self, rng, k, d):
        theta = rng.standard_normal(d)
        selected_indices = np.arange(k)
        theta_sparse = np.zeros(d)
        theta_sparse[selected_indices] = theta[selected_indices]
        theta = theta_sparse
        theta = theta / np.linalg.norm(theta)
        return theta

    def _generate_samples(self, Q, rng, d, m):
        X = rng.random((m, d))
        X_pair = rng.random((m, d))
        Delta_table = (X - X_pair) 

        Probs = Q.Pr_x1_less_x2(Delta_table)
        prefers = rng.random(m) < Probs
        prefer_mean = prefers * 2 - 1
        return Delta_table, prefer_mean

    def solve_MLE(self, method="SLSQP", options={"disp": False, "maxiter": 1000}):
        res_mle = minimize(
            loss_MLE,
            self.theta_init,
            args=(self.sigma, self.Delta_table, self.samples),
            method=method,
            constraints=self.constraints,
            options=options,
        )
        return res_mle, res_mle.x, self.dist(res_mle.x)**2, self.dist_sigma(res_mle.x)**2

    def solve_MLE_reg(
        self, method="trust-constr", options={"disp": False, "maxiter": 1000}
    ):
        res_mle = minimize(
            loss_MLE_reg,
            self.theta_init,
            args=(self.sigma, self.Delta_table, self.samples, self.beta),
            method=method,
            constraints=self.constraints,
            options=options,
        )
        return res_mle, res_mle.x, self.dist(res_mle.x)**2, self.dist_sigma(res_mle.x)**2

    def dist(self, theta):
        return np.linalg.norm(theta - self.theta)

    def dist_sigma(self, theta):
        return np.linalg.norm(self.Delta_table @ (theta - self.theta) / np.sqrt(self.m))


@click.group()
def cli():
    pass


##########################################
##########################################
@cli.command()
def exp1():
    data = []
    m = 100
    d = 100
    beta = 0.1
    sigma = 0.1
    for k in range(2, 101, 2):
        for seed in range(20):
            p = Problem(m=m, k=k, d=d, sigma=sigma, beta=beta, seed=seed)
            res_mlre, x_mlre, e_mlre, es_mlre = p.solve_MLE_reg()
            sparsity = np.sum(abs(x_mlre) > 1e-8) / d
            print(f"sparsity: {sparsity}")
            data.append(
                {
                    "m": m,
                    "k": k,
                    "d": d,
                    "sigma": sigma,
                    "beta": beta,
                    "seed": seed,
                    "e_mlre": e_mlre,
                    "es_mlre": es_mlre,
                    "res_mlre.success": res_mlre.success,
                    "res_mlre.status": res_mlre.status,
                    "res_mlre.message": res_mlre.message,
                    "res_mlre.nit": res_mlre.nit,
                    "sparsity": sparsity,
                }
            )
    data_df = pd.DataFrame(data)
    data_df.to_csv("data_csv_4fig12/exp1_sigma01_n100_seeds20.csv", index=False)


@cli.command()
def exp2():
    data = []
    d = 100
    k = 5
    sigma = 0.1
    for m in tqdm([10, 20, 40, 80, 100, 200, 400]):
        beta = 1 / (m ** 0.5)
        for seed in tqdm(range(20)):
            p = Problem(m=m, k=k, d=d, sigma=sigma, beta=beta, seed=seed)
            res_mle, x_mle, e_mle, es_mle = p.solve_MLE()
            res_mlre, x_mlre, e_mlre, es_mlre = p.solve_MLE_reg()
            data.append(
                {
                    "m": m,
                    "k": k,
                    "d": d,
                    "sigma": sigma,
                    "beta": beta,
                    "seed": seed,
                    "e_mle": e_mle,
                    "e_mlre": e_mlre,
                    "es_mle": es_mle,
                    "es_mlre": es_mlre,
                    "res_mle.success": res_mle.success,
                    "res_mlre.success": res_mlre.success,
                    "res_mle.status": res_mle.status,
                    "res_mlre.status": res_mlre.status,
                    "res_mle.message": res_mle.message,
                    "res_mlre.message": res_mlre.message,
                    "res_mle.nit": res_mle.nit,
                    "res_mlre.nit": res_mlre.nit,
                }
            )
    data_df = pd.DataFrame(data)
    data_df.to_csv("data_csv_4fig12/exp2_seeds20_beta05.csv", index=False)



@cli.command()
def exphyperopt():
    data = []
    d = 100
    k = 5
    sigma = 0.1
    print("sigma = ", sigma)
    print("k = ", k)

    def objective(beta, m):
        results = []
        for seed in range(10):
            p = Problem(m=m, k=k, d=d, sigma=sigma, beta=beta, seed=seed)
            res_mlre, x_mlre, e_mlre, es_mlre = p.solve_MLE_reg()
            results.append(
                {
                    "m": m,
                    "k": k,
                    "d": d,
                    "sigma": sigma,
                    "beta": beta,
                    "seed": seed,
                    "e_mlre": e_mlre,
                    "es_mlre": es_mlre,
                    "res_mlre.success": res_mlre.success,
                    "res_mlre.status": res_mlre.status,
                    "res_mlre.message": res_mlre.message,
                    "res_mlre.nit": res_mlre.nit,
                }
            )

        loss = np.mean([x["es_mlre"] for x in results])
        return {"loss": loss, "results": results, "status": "ok"}

    for m in tqdm([10, 20, 40, 50, 60, 70, 80, 90,  100, 200, 400, 800, 1000, 2000, 2500, 3000, 4000, 6000, 8000, 10000, 20000, 40000]): 
        objective_m = lambda beta: objective(beta, m)
        trials = Trials()
        space = hp.loguniform("beta", -10, 2)
        best = fmin(objective_m, space, algo=tpe.suggest, max_evals=20, trials=trials)
        filename = f"data_pkl_4beta_contour/sigma01_n{m}_2.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(trials, file)




if __name__ == "__main__":
    cli()
    cli.add_command(exp1)
    cli.add_command(exp2)
    cli.add_command(exphyperopt)
    
