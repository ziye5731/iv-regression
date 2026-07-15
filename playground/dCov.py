import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class QuadraticModel:
    def predict(self, theta, x):
        d_x = x.shape[-1]
        b = theta[:d_x].reshape(-1, 1)
        A = self._vec_to_mat(theta[d_x:], d_x)
        linear = x @ b
        quad = np.sum((x @ A) * x, axis=-1, keepdims=True)
        return linear + quad

    def gradient(self, theta, x):
        d_x = x.shape[-1]
        grad_b = x
        grad_full = np.einsum("ni,nj->nij", x, x)
        grad_packed_list = []
        for i in range(d_x):
            for j in range(i, d_x):
                if i == j:
                    grad_packed_list.append(grad_full[:, i, j:j+1])
                else:
                    grad_packed_list.append(2.0 * grad_full[:, i, j:j+1])
        grad_packed = np.concatenate(grad_packed_list, axis=-1)
        return np.concatenate([grad_b, grad_packed], axis=-1)

    def param_dim(self, d_x):
        return d_x + d_x * (d_x + 1) // 2

    @staticmethod
    def _vec_to_mat(vec, d_x):
        vec = np.atleast_1d(vec.ravel())
        A = np.zeros((d_x, d_x))
        idx = 0
        for i in range(d_x):
            for j in range(i, d_x):
                A[i, j] = vec[idx]
                if i != j:
                    A[j, i] = A[i, j]
                idx += 1
        return A

class OnlineDataGenerator:
    def __init__(self, d_x, d_z, theta_true, pi_strength=0.5, endog_corr=0.75, rng=None):
        self.d_x = d_x
        self.d_z = d_z
        self.theta_true = theta_true.reshape(-1, 1)
        self.model = QuadraticModel()
        self.rng = rng or np.random.default_rng()
        self.Pi = self.rng.normal(0, pi_strength / np.sqrt(d_z), size=(d_z, d_x))
        cov_v = np.eye(d_x)
        cov_cross = np.full((d_x, 1), endog_corr)
        cov_eps = np.array([[1.0]])
        self.full_cov = np.block([[cov_v, cov_cross],
                                  [cov_cross.T, cov_eps]])

    def next_sample(self):
        z = self.rng.normal(0, 1, size=(1, self.d_z))
        ve = self.rng.multivariate_normal(np.zeros(self.d_x + 1), self.full_cov, size=1)
        v = ve[:, :self.d_x]
        eps = ve[:, self.d_x:]
        x = z @ self.Pi + v
        y = self.model.predict(self.theta_true, x) + eps
        return z, x, y

def warm_start_from_batch(Z, X, Y, model):
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Z, X)
    X_hat = lr.predict(Z)
    n, d = X_hat.shape
    phi = [X_hat]
    for i in range(d):
        for j in range(i, d):
            if i == j:
                col = (X_hat[:, i] * X_hat[:, j]).reshape(-1, 1)
            else:
                col = (2.0 * X_hat[:, i] * X_hat[:, j]).reshape(-1, 1)
            phi.append(col)
    phi = np.concatenate(phi, axis=1)
    theta_flat = np.linalg.lstsq(phi, Y.ravel(), rcond=None)[0]
    return theta_flat.reshape(-1, 1)

def _one_quadruple_gradient(Z, X, Y, theta, model):
    d_theta = model.param_dim(X.shape[1])
    e = Y - model.predict(theta.reshape(-1,1), X)
    grad_g = model.gradient(theta.reshape(-1,1), X)
    a = cdist(Z, Z, 'euclidean')
    grad_b = np.zeros((4, 4, d_theta))
    for i in range(4):
        for j in range(4):
            if i == j: continue
            diff = e[i] - e[j]
            norm = np.abs(diff)
            if norm > 1e-12:
                coeff = diff / norm
                grad_b[i, j] = coeff * (grad_g[j] - grad_g[i])
    term1 = np.zeros(d_theta)
    for i in range(4):
        for j in range(i+1, 4):
            term1 += a[i, j] * grad_b[i, j]
    term2 = np.zeros(d_theta)
    for i in range(4):
        for j in range(4):
            if j == i: continue
            for k in range(4):
                if k == i or k == j: continue
                term2 += a[i, j] * grad_b[i, k]
    term3 = np.zeros(d_theta)
    partitions = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]
    for (i,j), (k,l) in partitions:
        term3 += a[i, j] * grad_b[k, l]
    return term1/6.0 - term2/12.0 + term3/3.0

def run_online_sgd(batch_size=128, lr=0.005, n_repeats=10, n_updates=10000, warmup_samples=200):
    d_x = 4
    d_z = 8
    pi_strength = 0.5
    endog_corr = 0.75
    # 自动设定 num_quadruples = batch_size * 4
    num_quadruples = batch_size * 4

    model = QuadraticModel()
    rng_true = np.random.default_rng(12345)
    theta_true = np.zeros((model.param_dim(d_x), 1))
    theta_true[:d_x, 0] = rng_true.normal(0, 1.0, size=d_x)
    n_quad = model.param_dim(d_x) - d_x
    if n_quad > 0:
        nz = max(1, n_quad // 3)
        idx = rng_true.choice(n_quad, size=nz, replace=False)
        theta_true[d_x + idx, 0] = rng_true.normal(0, 0.5, size=nz)
    theta_true_flat = theta_true.ravel()

    all_dist_hist = []

    for rep in range(n_repeats):
        rng = np.random.default_rng(42 + rep * 100)
        data_gen = OnlineDataGenerator(d_x, d_z, theta_true, pi_strength, endog_corr, rng)

        Z_warm, X_warm, Y_warm = [], [], []
        for _ in range(warmup_samples):
            z, x, y = data_gen.next_sample()
            Z_warm.append(z)
            X_warm.append(x)
            Y_warm.append(y)
        Z_warm = np.vstack(Z_warm)
        X_warm = np.vstack(X_warm)
        Y_warm = np.vstack(Y_warm)
        theta = warm_start_from_batch(Z_warm, X_warm, Y_warm, model).ravel()

        dist_history = [np.linalg.norm(theta - theta_true_flat)]

        for step in range(n_updates):
            Z_batch = np.zeros((batch_size, d_z))
            X_batch = np.zeros((batch_size, d_x))
            Y_batch = np.zeros((batch_size, 1))
            for i in range(batch_size):
                z, x, y = data_gen.next_sample()
                Z_batch[i] = z
                X_batch[i] = x
                Y_batch[i] = y

            grad_est = np.zeros(model.param_dim(d_x))
            for _ in range(num_quadruples):
                idx = np.random.choice(batch_size, size=4, replace=False)
                grad_est += _one_quadruple_gradient(
                    Z_batch[idx], X_batch[idx], Y_batch[idx], theta, model)
            grad_est /= num_quadruples

            theta = theta - lr * grad_est

            if (step+1) % 10 == 0:
                dist_history.append(np.linalg.norm(theta - theta_true_flat))

        all_dist_hist.append(dist_history)
        print(f"Repeat {rep+1}/{n_repeats}, final dist = {dist_history[-1]:.4f}")

    # 绘图
    max_len = max(len(h) for h in all_dist_hist)
    dist_matrix = np.full((n_repeats, max_len), np.nan)
    for i, h in enumerate(all_dist_hist):
        dist_matrix[i, :len(h)] = h

    steps = np.arange(max_len) * 10
    median = np.nanmedian(dist_matrix, axis=0)
    q25 = np.nanpercentile(dist_matrix, 25, axis=0)
    q75 = np.nanpercentile(dist_matrix, 75, axis=0)

    plt.figure(figsize=(10, 6))
    for i in range(n_repeats):
        plt.plot(steps[:len(all_dist_hist[i])], all_dist_hist[i],
                 color='gray', alpha=0.2, linewidth=0.8)
    plt.plot(steps, median, color='blue', linewidth=2, label='Median')
    plt.fill_between(steps, q25, q75, color='blue', alpha=0.2, label='25%–75% quantile')
    plt.xlabel('SGD update (each uses a batch of new samples)')
    plt.ylabel('Euclidean distance to true parameters')
    plt.title(f'U‑statistic dCov² SGD (batch size={batch_size}, auto quadruples={num_quadruples}, lr={lr})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 只需调节 batch_size
    run_online_sgd(batch_size=16, lr=0.005)