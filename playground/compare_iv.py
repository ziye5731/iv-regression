import torch
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)



# ============================================================
# True nonlinear IV model
#
# X = h(Z) + eta
# Y = g_theta(X) + epsilon
#
# epsilon correlated with eta
# but independent of Z
# ============================================================


theta_true = torch.tensor([1.0, 0.5])


# structural equation
def g(theta, x):
    """
    g_theta(x)=theta1*x+theta2*x^2
    """
    return theta[0]*x + theta[1]*x**2



def grad_g(theta, x):
    """
    gradient wrt theta
    """
    return torch.stack(
        [
            x,
            x**2
        ]
    )



# nonlinear first stage
def h(gamma, z):

    return (
        gamma[0]*z
        +
        gamma[1]*torch.sin(z)
    )



def grad_h(gamma,z):

    return torch.stack(
        [
            z,
            torch.sin(z)
        ]
    )



# ============================================================
# Data generator
# ============================================================

def sample():

    z = torch.randn(())

    eta = torch.randn(())


    x = (
        2*z
        +
        torch.sin(z)
        +
        eta
    )


    # confounding:
    # epsilon correlated with eta
    u=torch.randn(())

    eps = 0.8*eta + 0.5*u


    y = g(theta_true,x)+eps


    return z,x,y



# ============================================================
# 1. Online nonlinear 2SLS
# ============================================================


def run_2sls(T):


    theta=torch.zeros(2)

    gamma=torch.zeros(2)


    lr1=0.01
    lr2=0.005


    errors=[]


    for t in range(T):


        z,x,y=sample()



        # ---------------------
        # first stage
        # X=h_gamma(Z)
        # ---------------------

        xhat=h(gamma,z)


        loss=x-xhat


        grad_gamma=-2*loss*grad_h(gamma,z)


        gamma -= lr1*grad_gamma



        # ---------------------
        # second stage
        # Y=g_theta(Xhat)
        # ---------------------

        xhat=h(gamma,z)

        pred=g(theta,xhat)


        grad_theta=(
            2*(pred-y)
            *
            grad_g(theta,xhat)
        )


        theta -= lr2*grad_theta



        errors.append(
            torch.norm(theta-theta_true).item()
        )


    return errors




# ============================================================
# 2. Online GMM
#
# E[Z(Y-g_theta(X))]=0
# ============================================================


def run_gmm(T):


    theta=torch.zeros(2)


    lr=0.005


    errors=[]



    for t in range(T):


        z,x,y=sample()


        eps=y-g(theta,x)


        moment=z*eps



        grad_m=(
            -z
            *
            grad_g(theta,x)
        )


        grad=moment*grad_m



        theta -= lr*grad



        errors.append(
            torch.norm(theta-theta_true).item()
        )


    return errors





# ============================================================
# 3. dCov optimization
#
# minimize dCov(Z, epsilon_theta)
#
# epsilon_theta=Y-g_theta(X)
#
# ============================================================



def dcov_gradient(Z,E,theta,X):


    n=len(Z)


    grad=torch.zeros(2)



    # simplified U-statistic gradient

    for i in range(n):

        for j in range(n):

            if i==j:
                continue


            dz=torch.abs(
                Z[i]-Z[j]
            )


            diff=E[i]-E[j]


            sign=torch.sign(diff)



            grad_eps=(
                -grad_g(theta,X[i])
                +
                grad_g(theta,X[j])
            )


            grad += (
                dz
                *
                sign
                *
                grad_eps
            )



    return grad/(n*n)





def run_dcov(T,batch_size=32):


    theta=torch.zeros(2)


    lr=0.001


    errors=[]



    for t in range(T):


        Z=[]
        X=[]
        Y=[]


        for _ in range(batch_size):

            z,x,y=sample()

            Z.append(z)
            X.append(x)
            Y.append(y)



        Z=torch.stack(Z)

        X=torch.stack(X)

        Y=torch.stack(Y)



        eps=Y-g(theta,X)



        grad=dcov_gradient(
            Z,
            eps,
            theta,
            X
        )



        theta -= lr*grad



        errors.append(
            torch.norm(theta-theta_true).item()
        )


    return errors





# ============================================================
# Run experiments
# ============================================================


if __name__=="__main__":


    T=5000



    print("Running nonlinear online 2SLS...")
    e_2sls=run_2sls(T)



    print("Running nonlinear GMM...")
    e_gmm=run_gmm(T)



    print("Running dCov optimization...")
    e_dcov=run_dcov(T)



    # ----------------------------
    # plot
    # ----------------------------

    plt.figure(figsize=(8,5))


    plt.semilogy(
        e_2sls,
        label="Online nonlinear 2SLS"
    )


    plt.semilogy(
        e_gmm,
        label="Online GMM"
    )


    plt.semilogy(
        e_dcov,
        label="dCov minimization"
    )



    plt.xlabel(
        "Iteration"
    )


    plt.ylabel(
        r"$||\theta-\theta^*||_2$"
    )


    plt.title(
        "Nonlinear Online IV Regression"
    )


    plt.grid(True)


    plt.legend()



    plt.show()