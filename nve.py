import torch
import matplotlib.pyplot as plt
from pot import compute_Muller_potential


def Molecular_Dynamics_sampling(
    x1_min,
    x1_max,
    x2_min,
    x2_max,
    scale=0.05,
    num_reps=10,
    num_steps=110000,
    dt=0.001,
    cal_bias=False,
):
    # 初始化参数
    n_particles = num_reps
    n_steps = num_steps
    dt = dt
    mass = 1.0

    # 随机初始化位置和速度
    x1 = (
        torch.rand((n_particles), dtype=torch.float32) * (x1_max - x1_min) + x1_min
    )  # 初始化原子位置
    x2 = torch.rand((n_particles), dtype=torch.float32) * (x2_max - x2_min) + x2_min
    x = torch.stack((x1, x2), dim=1)
    v = torch.randn((n_particles, 2), dtype=torch.float32) * 0.0001
    # 使x可计算梯度
    x.requires_grad_(True)
    if cal_bias:
        energy = compute_Muller_potential(scale, x, cal_bias)
    else:
        energy = compute_Muller_potential(scale, x, cal_bias=False)

    # 计算初始力和加速度
    force = -torch.autograd.grad(
        outputs=energy,
        inputs=x,
        grad_outputs=torch.ones_like(energy),
        create_graph=True,
        retain_graph=True,
    )[0]
    a = force / mass

    # 存储轨迹
    trajectory = [x.clone().detach()]

    # Velocity Verlet 算法
    for step in range(n_steps):
        if (step + 1) % 10000 == 0:
            print("steps: {} out of {} total steps".format(step + 1, num_steps))

        # 更新位置
        x = x + v * dt + 0.5 * a * dt**2
        x.requires_grad_(True)  # 使新位置可计算梯度

        # 检查是否有粒子超出了边界，并反转其速度
        for i in range(n_particles):
            if x[i, 0] < x1_min or x[i, 0] > x1_max:
                v[i, 0] = -v[i, 0]  # 反转x方向速度
                x[i, 0] = torch.clamp(x[i, 0], x1_min, x1_max)  # 限制位置在边界内

            if x[i, 1] < x2_min or x[i, 1] > x2_max:
                v[i, 1] = -v[i, 1]  # 反转y方向速度
                x[i, 1] = torch.clamp(x[i, 1], x2_min, x2_max)  # 限制位置在边界内

        # 计算新的力和加速度
        if cal_bias:
            energy_new = compute_Muller_potential(scale, x, cal_bias)
        else:
            energy_new = compute_Muller_potential(scale, x, cal_bias=False)

        # 计算初始力和加速度
        force_new = -torch.autograd.grad(
            outputs=energy_new,
            inputs=x,
            grad_outputs=torch.ones_like(energy_new),
            create_graph=True,
            retain_graph=True,
        )[0]
        a_new = force_new / mass

        # 更新速度
        v = v + 0.5 * (a + a_new) * dt

        # 更新加速度
        a = a_new
        if (step % 100 == 0) & (step >= 10000):
            # 存储轨迹
            trajectory.append(x.clone().detach())

    # 转换为 NumPy 数组以便绘图
    trajectory = torch.stack(trajectory).numpy()

    return trajectory


if __name__ == "__main__":
    x1_min, x1_max = -1.5, 1.0
    x2_min, x2_max = -0.5, 2.0
    fig, axes = plt.subplots()
    scale = 0.05

    ## draw samples from the Müller potential using temperature replica exchange
    ## Molecular Dynamics sampling
    ############################################################################

    num_reps = 10  # number of replicas
    num_steps = 110000
    trajectory = Molecular_Dynamics_sampling(
        x1_min, x1_max, x2_min, x2_max, scale, num_reps, num_steps
    )

    #### plot samples
    fig = plt.figure()
    fig.clf()
    x1_vals = trajectory[:, :, 0].flatten()
    x2_vals = trajectory[:, :, 1].flatten()
    plt.plot(x1_vals, x2_vals, ".", alpha=0.5)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel(r"$x_1$", fontsize=24)
    plt.ylabel(r"$x_2$", fontsize=24)
    axes.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        "/data/home/yfsun/python_code/Muller_Brown_pot_sampling/mpnvesampling.png"
    )
    plt.show()
