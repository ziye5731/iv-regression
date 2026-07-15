import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

# ==================== 1. 数据生成 ====================
def generate_data(n, beta_true=1.0, gamma=0.8, rho=0.6, seed=None):
    """
    生成线性 IV 模型数据，带有异方差误差。
    模型:
        y = X * beta + e
        X = Z * gamma + v
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.normal(0, 1, size=(n, 1))
    v = np.random.normal(0, 1, size=(n, 1))
    xi = np.random.normal(0, 1, size=(n, 1))
    e_base = rho * v + np.sqrt(1 - rho**2) * xi
    hetero_scale = 0.5 + 0.3 * np.abs(Z)
    e = hetero_scale * e_base
    X = gamma * Z + v
    y = X * beta_true + e
    return y, X, Z

# ==================== 2. 估计方法（全部返回 float） ====================
def estimate_2sls(y, X, Z):
    """两阶段最小二乘，返回标量估计"""
    Z_with_const = sm.add_constant(Z)
    stage1 = sm.OLS(X, Z_with_const).fit()
    X_hat = stage1.predict()
    X_hat_with_const = sm.add_constant(X_hat)
    stage2 = sm.OLS(y, X_hat_with_const).fit()
    return float(stage2.params[1])          # 转换为 float

def estimate_gmm(y, X, Z):
    """两步线性 GMM（异方差稳健权重），返回标量估计"""
    n = X.shape[0]
    Z_mat = np.hstack([np.ones((n, 1)), Z])
    # 第一步
    W1 = np.linalg.inv(Z_mat.T @ Z_mat / n)
    M_zz = Z_mat.T @ X / n
    M_zy = Z_mat.T @ y / n
    beta_init = np.linalg.solve(M_zz.T @ W1 @ M_zz, M_zz.T @ W1 @ M_zy)
    # 第二步
    e_hat = y - X @ beta_init
    S = (Z_mat * e_hat).T @ (Z_mat * e_hat) / n
    W2 = np.linalg.inv(S)
    beta_gmm = np.linalg.solve(M_zz.T @ W2 @ M_zz, M_zz.T @ W2 @ M_zy)
    return float(beta_gmm[0])               # 转换为 float

def distance_covariance_sqr(u, v):
    """计算残差 u 与工具变量 v 的平方距离协方差"""
    n = u.shape[0]
    a = np.abs(u - u.T)
    b = np.sqrt(np.sum((v[:, np.newaxis, :] - v[np.newaxis, :, :])**2, axis=2))
    A = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    B = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    return np.sum(A * B) / (n * n)

def estimate_dcov_min(y, X, Z, beta_init=None):
    """最小化距离协方差，返回标量估计"""
    if beta_init is None:
        beta_init = estimate_2sls(y, X, Z)
    def objective(beta):
        e = y - X * beta
        return distance_covariance_sqr(e, Z)
    result = minimize(objective, beta_init, method='BFGS')
    return float(result.x[0])               # 转换为 float

# ==================== 3. Monte Carlo 模拟 ====================
def run_simulation(n, N, beta_true=1.0):
    estimates = {'2SLS': [], 'GMM': [], 'dCov': []}
    for i in range(N):
        y, X, Z = generate_data(n, beta_true, seed=i)
        estimates['2SLS'].append(estimate_2sls(y, X, Z))
        estimates['GMM'].append(estimate_gmm(y, X, Z))
        estimates['dCov'].append(estimate_dcov_min(y, X, Z))
        if (i+1) % 20 == 0:
            print(f"完成 {i+1}/{N} 次模拟...")
    return estimates

# ==================== 4. 结果分析与可视化 ====================
def analyze_results(estimates, beta_true):
    df = pd.DataFrame(estimates)
    methods = df.columns
    stats = pd.DataFrame(index=methods, columns=['Mean', 'Bias', 'Std', 'RMSE'])
    for m in methods:
        stats.loc[m, 'Mean'] = df[m].mean()
        stats.loc[m, 'Bias'] = df[m].mean() - beta_true
        stats.loc[m, 'Std'] = df[m].std()
        stats.loc[m, 'RMSE'] = np.sqrt(np.mean((df[m] - beta_true)**2))
    print("\n===== 估计结果比较 =====")
    print(f"真实参数 beta = {beta_true}")
    print(stats.to_string(float_format=lambda x: "{:.4f}".format(x)))

    plt.figure(figsize=(8, 5))
    df.boxplot(column=methods, grid=False)
    plt.axhline(y=beta_true, color='r', linestyle='--', label=f'True beta={beta_true}')
    plt.title('参数估计分布 (2SLS vs GMM vs min dCov)')
    plt.ylabel('估计值')
    plt.legend()
    plt.tight_layout()
    plt.savefig('estimation_comparison.png')
    plt.show()
    return df, stats

# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    np.random.seed(123)
    n_sample = 200
    N_sim = 200
    true_beta = 1.5

    print(f"开始模拟: 样本量={n_sample}, 模拟次数={N_sim}, 真实beta={true_beta}")
    estimates = run_simulation(n_sample, N_sim, true_beta)
    df, stats = analyze_results(estimates, true_beta)

    df.to_csv('estimates.csv', index=False)
    stats.to_csv('estimation_stats.csv')