import numpy as np
import scipy.linalg
from scipy.sparse import diags
import matplotlib.pyplot as plt
import sys


# 1D model problem: -u_xx = f
# with homogeneous boundary conditions (BC).

# Source term
def f(x, k):
    return k * k * np.pi * np.pi * np.sin(k * np.pi * x)


# Exact solution
def u_ex(x, k):
    return np.sin(k * np.pi * x)


def gen_linsys(nx, dx, u, f_arr):
    diag_a = -np.ones(nx)
    diag_c = 2 * np.ones(nx)
    diagonals = [diag_a, diag_c, diag_a]
    offsets = [-1, 0, 1]
    boundary = [0, nx - 1]
    interior = np.arange(1, nx - 1)
    mat_a = diags(diagonals, offsets, (nx, nx)).toarray()
    f_arr *= dx * dx
    mat_a[np.ix_(boundary, boundary)] = np.eye(len(boundary))
    s = mat_a[np.ix_(interior, boundary)] @ u[boundary]
    f_arr[interior] -= s
    mat_a[np.ix_(interior, boundary)] = 0
    mat_a[np.ix_(boundary, interior)] = 0
    return mat_a, f_arr


def main():
    maxiter = 8
    check_freq = 2
    ncheck = maxiter // check_freq + 1
    # Grid parameters.
    nx = 2**7 + 1  # number of POINTS in the x direction
    k = nx // 4
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    lx = xmax - xmin  # domain length in the x direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    x = np.linspace(xmin, xmax, nx)
    # analytical solution is high-freq + low-freq
    u_analytic = u_ex(x, k) + 10 * u_ex(x, 1)
    f_arr = f(x, k) + 10 * f(x, 1)
    # generate linear system and apply dirichlet BC
    A, f_arr = gen_linsys(nx, dx, u_analytic, f_arr)
    # exact numerical solution
    u_exact = scipy.linalg.solve(A, f_arr)
    print(f"Discretization Error: {np.linalg.norm(u_exact-u_analytic, ord=np.inf):.4e}")
    fig, ax = plt.subplots(1)
    ax.plot(x, u_analytic, marker='o')
    ax.plot(x, u_exact, 'r', alpha=0.2)
    ax.set_title("analytical and numerical solutions of -u_{xx}=f")

    # Gauss-Seidel iterations
    L = np.tril(A)
    u = np.zeros(nx)
    fig, ax = plt.subplots(ncheck + 1)
    fig2, ax2 = plt.subplots(ncheck)
    ax[0].plot(x, u_analytic, linestyle='-')
    ax_i = 0
    for i in range(maxiter):
        e = u_exact - u
        r = f_arr - A @ u
        u = u + scipy.linalg.solve_triangular(L, r, lower=True)
        print(f"Iteration {i:6d}/{maxiter:6d}, residual: {np.linalg.norm(r):.4e}, "
              f"Solution err 2-norm: {np.linalg.norm(e):.4e}, "
              f"A-norm: {e.transpose() @ A @ e:4e}")
        if np.remainder(i, check_freq) == 0 or i == maxiter - 1:
            ax[ax_i+1].plot(x, u, linestyle='-')
            ax2[ax_i].plot(x, abs(e), linestyle='-')
            ax_i += 1

    for i in range(len(ax)-1):
        ax[i].get_xaxis().set_visible(False)
    for i in range(len(ax2)-1):
        ax2[i].get_xaxis().set_visible(False)

    ax[0].set_title("iterative solution: u")
    ax2[0].set_title("error: u_exact - u")

    return 0


if __name__ == "__main__":
    err = main()
    plt.show()
    sys.exit(err)
