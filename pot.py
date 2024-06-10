import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from typing import Optional


def compute_Muller_potential(
    scale,
    x,
    cal_bias=False,
    center: Optional[Tensor] = None,
    x1_min: Optional[float] = None,
    x1_max: Optional[float] = None,
    x2_min: Optional[float] = None,
    x2_max: Optional[float] = None,
    ifmc: Optional[bool] = None
):
    A = (-200.0, -100.0, -170.0, 15.0)
    beta = (0.0, 0.0, 11.0, 0.6)
    alpha_gamma = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5, -6.5]),
        x.new_tensor([0.7, 0.7]),
    )

    ab = (
        x.new_tensor([1.0, 0.0]),
        x.new_tensor([0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0]),
    )

    U = 0
    for i in range(4):
        diff = x - ab[i]
        U = U + A[i] * torch.exp(
            torch.sum(alpha_gamma[i] * diff**2, -1) + beta[i] * torch.prod(diff, -1)
        )

    U = scale * U
    if cal_bias:
        for icenter in center:
            if ifmc:
                height = 50
                bias = compute_bias(icenter, x, x1_min, x1_max, x2_min, x2_max, height)
            else:
                bias = compute_bias(icenter, x, x1_min, x1_max, x2_min, x2_max)
            U += bias
    return U


def compute_bias(center, x, x1_min, x1_max, x2_min, x2_max, height=1.0):
    """
    根据轨迹数据计算高斯偏置势能
    """
    bias = 0.0
    # 定义标准差
    sigma_x1 = (x1_max - x1_min) / 6  # 假设 3 sigma 覆盖整个范围
    sigma_x2 = (x2_max - x2_min) / 6  # 假设 3 sigma 覆盖整个范围
    # 计算高斯函数值
    bias = height * torch.exp(
        -(
            (x[:, 0] - center[0]) ** 2 / (2 * sigma_x1**2)
            + (x[:, 1] - center[1]) ** 2 / (2 * sigma_x2**2)
        )
    )
    return bias


def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))  ## -1 自己推断未指定的值
    return x


if __name__ == "__main__":
    x1_min, x1_max = -1.5, 1.0
    x2_min, x2_max = -0.5, 2.0

    grid_size = 100
    x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, grid_size)
    fig, axes = plt.subplots()
    scale = 0.05
    U = compute_Muller_potential(scale, x_grid)
    U = U.reshape(100, 100)
    U[U > 9] = 9
    U = U.T
    plt.contourf(
        U,
        levels=np.linspace(-9, 9, 19),
        extent=(x1_min, x1_max, x2_min, x2_max),
        cmap=cm.viridis_r,
    )
    plt.xlabel(r"$x_1$", fontsize=24)
    plt.ylabel(r"$x_2$", fontsize=24)
    plt.colorbar()
    axes.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("/data/home/yfsun/python_code/Muller_Brown_pot_sampling/mp1.png")
    plt.close()
