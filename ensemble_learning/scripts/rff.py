"""
Gradient Boosting with Random Fourier Features (RFF)

"""

import numpy as np
from scipy import optimize
from scipy.optimize import fmin_l_bfgs_b
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

##################################################################
########### GradientBoostRFF

class GradientBoostingRFF:
    def __init__(self, T=20, gamma=1.0, lambda_reg=1.0, maxiter_xt=200, maxiter_omega=200,
                 verbose=False):
        """
        Parameters:
            T: int, nombre d'itérations (d'itérations faibles car chaque itération fait deux optimizations)
            gamma: float, paramètre pour l'échantillonnage initial de omega ~ N(0, 2*gamma I)
            lambda_reg: float, régularisation sur ||omega||^2 dans l'étape 7
            maxiter_xt, maxiter_omega: int, limites d'itérations pour fmin_l_bfgs_b
            verbose: bool, affichage d'informations pendant l'apprentissage
        """
        self.T = T
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.maxiter_xt = maxiter_xt
        self.maxiter_omega = maxiter_omega
        self.verbose = verbose

        # stockage des paramètres appris
        self.Xt = []        # liste de vecteurs x_t (d,)
        self.Omegat = []    # liste de vecteurs omega_t (d,)
        self.alphat = []    # liste de scalaires alpha_t

    def _H(self, X):
        """Calcule H(x) pour une matrice X (m x d) avec les paramètres stockés."""
        if len(self.Omegat) == 0:
            return np.zeros(X.shape[0])
        H = np.zeros(X.shape[0])
        for alpha, omega, xt in zip(self.alphat, self.Omegat, self.Xt):
            # compute cos(omega . (x - xt)) for all x in X
            proj = X.dot(omega) - np.dot(omega, xt)
            H += alpha * np.cos(proj)
        return H

    def predict(self, X):
        H = self._H(np.atleast_2d(X))
        return np.sign(H)

    def fit(self, X, y):
        """
        Apprend le modèle à partir des données X (n,d) et labels y (n,) dans {-1,+1}.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        n, d = X.shape
        assert y.shape[0] == n
        assert set(np.unique(y)).issubset({-1.0, 1.0})

        # initial H (H0)
        H = 0.5 * np.log(np.sum(1 + y) / np.sum(1 - y))

        for t in range(self.T):
            # 3: poids wi = exp(- y_i * H_{t-1}(x_i))
            w = np.exp(- y * H)

            # 4: ytilde = y * w
            ytilde = y * w

            # 5: draw initial omega ~ N(0, 2*gamma I)
            omega0 = np.random.normal(loc=0.0, scale=np.sqrt(2.0 * self.gamma), size=d)

            # 6: optimise xt given omega0
            def obj_xt(xt_flat):
                xt = xt_flat.reshape(d)
                u = X.dot(omega0) - np.dot(omega0, xt)  # shape (n,)
                ex = np.exp(- ytilde * np.cos(u))
                val = ex.mean()
                return val

            def grad_xt(xt_flat):
                xt = xt_flat.reshape(d)
                u = X.dot(omega0) - np.dot(omega0, xt)
                ex = np.exp(- ytilde * np.cos(u))
                # derivative wrt xt: (1/n) * sum ex * (-ytilde) * omega0 * sin(u)
                coeff = (- ytilde * ex).reshape(-1, 1)  # shape (n,1)
                # sum over i of coeff_i * sin(u_i)
                s = np.sum(coeff.flatten() * np.sin(u))
                # but we must multiply by omega0 (vector). However existing expression is scalar times omega0.
                grad = (1.0 / n) * omega0 * s
                return grad

            # initial guess for xt: sample mean
            xt0 = X.mean(axis=0)
            if self.verbose:
                print(f"Iteration {t+1}/{self.T}: optimizing x_t (init from mean)")

            xt_opt, _, _ = fmin_l_bfgs_b(func=obj_xt, x0=xt0, fprime=grad_xt,
                                         approx_grad=False,
                                         bounds=[(None, None)] * d,
                                         maxiter=self.maxiter_xt)
            xt_opt = xt_opt.reshape(d)

            # 7: optimise omega with regularisation using xt_opt as fixed
            def obj_omega(omega_flat):
                omega = omega_flat.reshape(d)
                u = X.dot(omega) - np.dot(omega, xt_opt)
                ex = np.exp(- ytilde * np.cos(u))
                val = 0.5 * self.lambda_reg * np.dot(omega, omega) + ex.mean()
                return val

            def grad_omega(omega_flat):
                omega = omega_flat.reshape(d)
                u = X.dot(omega) - np.dot(omega, xt_opt)
                ex = np.exp(- ytilde * np.cos(u))
                # derivative of ex term: (1/n) sum ytilde_i * ex_i * sin(u_i) * (x_i - xt)
                coeff = (ytilde * ex).reshape(-1, 1)  # shape (n,1)
                sin_u = np.sin(u).reshape(-1, 1)
                # each row contributes coeff_i * sin(u_i) * (x_i - xt)
                diff = X - xt_opt.reshape(1, -1)  # shape (n,d)
                grad_ex = np.sum(coeff * sin_u * diff, axis=0) / n
                grad = self.lambda_reg * omega + grad_ex
                return grad

            if self.verbose:
                print(f"Iteration {t+1}/{self.T}: optimizing omega_t (init from normal)")

            omega_opt, _, _ = fmin_l_bfgs_b(func=obj_omega, x0=omega0, fprime=grad_omega,
                                            approx_grad=False,
                                            bounds=[(None, None)] * d,
                                            maxiter=self.maxiter_omega)
            omega_opt = omega_opt.reshape(d)

            # 8: compute alpha_t via the formula
            proj = X.dot(omega_opt) - np.dot(omega_opt, xt_opt)
            c = np.cos(proj)
            # numerator = sum_i (1 + y_i * c_i) * w_i
            num = np.sum((1.0 + y * c) * w)
            den = np.sum((1.0 - y * c) * w)
            # numerical stability: clip
            eps = 1e-12
            num = np.maximum(num, eps)
            den = np.maximum(den, eps)
            alpha_t = 0.5 * np.log(num / den)

            # 9: update H
            H = H + alpha_t * np.cos(proj)

            # store parameters
            self.Xt.append(xt_opt.copy())
            self.Omegat.append(omega_opt.copy())
            self.alphat.append(float(alpha_t))

            if self.verbose:
                print(f"  alpha_{t+1} = {alpha_t:.5f}, ||omega||={np.linalg.norm(omega_opt):.5f}")

        # fin de l'apprentissage
        return self

##################################################################
########### Gradient RFF Boosting   

def make_gradient_boosting_rff(params):

    """
    Retourne le modèle de boosting basé sur le RFF.
    """

    return GradientBoostingRFF(**params)


##################################################################
########### XGBoost

def make_xgboost(T):

    """
    Retourne le modèle XGBoost
    """

    return XGBClassifier(n_estimators=list(T.values())[0])
 
##################################################################
########### LGBM

def make_lgbm(T): 

    """
    Retourne le modèle LGBM
    """
    return LGBMClassifier(n_estimators=list(T.values())[0], verbose=-1)