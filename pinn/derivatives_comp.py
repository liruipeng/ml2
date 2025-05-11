import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys

class ProbParam:
    def __init__(self):
        self.w = [1, 2, 4, 8, 16]
        self.c = [1., 1., 1., 1., 1.]

prob = ProbParam()

# Define the function for which you want to compute the derivative
def my_function(x):
    lw = len(prob.w)
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * torch.sin(w * np.pi * x)
    return y

# Second derivative
def my_function_2der(x):
    lw = len(prob.w)
    y = torch.zeros_like(x)
    for i in range(lw):
        w = prob.w[i]
        c = prob.c[i]
        y += c * (w * w * np.pi * np.pi) * torch.sin(w * np.pi * x)
    return (-y)

# Points at which you want to compute the derivative
# You can replace this with your own set of points
ax = 0.0
bx = 1.0
ntrain = 30
x_train_np = np.linspace(ax, bx, ntrain)[:, None][1:-1]
x_train = torch.tensor(x_train_np, requires_grad=True)


# Compute the derivative of the function at the given points

u = my_function(x_train) #.clone().detach().requires_grad_(True)
du_dx_ad = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), create_graph=True)
du_dx_ad = du_dx_ad[0]
d2u_dx2_ad = torch.autograd.grad(du_dx_ad, x_train, grad_outputs=torch.ones_like(du_dx_ad), create_graph=True)
d2u_dx2_ad = d2u_dx2_ad[0]

h = 1e-5
h_v = h * torch.ones_like(x_train)
u = my_function(x_train)
u_plus = my_function(x_train + h_v)
u_minus = my_function(x_train - h_v)

d2u_dx2_fd = (u_plus[:, 0] + u_minus[:, 0] - 2 * u[:, 0]) / (h ** 2)
d2u_dx2_fd = (u_plus[:, 0] + u_minus[:, 0] - 2 * u[:, 0]) / (h ** 2)

d2u_dx2_fd = d2u_dx2_fd.unsqueeze(-1)


d2u_dx2 = my_function_2der(x_train)

# Print the results
print(f"FD-Autograd error: {torch.norm(d2u_dx2_fd - d2u_dx2_ad)/torch.norm(d2u_dx2_ad):.4e} ")
print(f"FD-Exact error: {torch.norm(d2u_dx2_fd - d2u_dx2)/torch.norm(d2u_dx2):.4e} ")
print(f"Autograd-Exact error: {torch.norm(d2u_dx2_ad - d2u_dx2)/torch.norm(d2u_dx2):.4e} ")
      
#for x, gradient in zip(x_train, gradients):
#    print(f"Point: {x.item()}, Derivative: {gradient.item()}")