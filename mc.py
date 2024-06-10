import torch
import matplotlib.pyplot as plt
import numpy as np
from pot import compute_Muller_potential

def Monte_Carlo_sampling(x1_min, x1_max, x2_min, x2_max, scale=0.05, num_reps=10, num_steps=110000, cal_bias=False):
    x_record = []
    accept_rate = 0
    scales = torch.linspace(0.0, scale, num_reps)
    x = torch.stack((x1_min + torch.rand(num_reps)*(x1_max - x1_min),
                     x2_min + torch.rand(num_reps)*(x2_max - x2_min)),
                    dim = -1)
    if cal_bias:
        energy = compute_Muller_potential(1.0, x, cal_bias)
    else:
        energy = compute_Muller_potential(1.0, x, cal_bias=False)
    for k in range(num_steps):
        if (k + 1) % 10000 == 0:
            print("steps: {} out of {} total steps".format(k+1, num_steps))

        ## sampling within each replica
        delta_x = torch.normal(0, 1, size = (num_reps, 2))*0.3
        x_p = x + delta_x
        if cal_bias:
            energy_p = compute_Muller_potential(1.0, x_p, cal_bias)
        else:
            energy_p = compute_Muller_potential(1.0, x_p, cal_bias=False)
        ## accept based on energy
        accept_prop = torch.exp(-scales*(energy_p - energy))
        accept_flag = torch.rand(num_reps) < accept_prop

        ## considering the bounding effects
        accept_flag = accept_flag & torch.all(x_p > x_p.new_tensor([x1_min, x2_min]), -1) \
                                  & torch.all(x_p < x_p.new_tensor([x1_max, x2_max]), -1)

        x_p[~accept_flag] = x[~accept_flag]
        energy_p[~accept_flag] = energy[~accept_flag]
        x = x_p
        energy = energy_p

        ## calculate overall accept rate
        accept_rate = accept_rate + (accept_flag.float() - accept_rate)/(k+1)

        ## exchange
        if k % 10 == 0:
            for i in range(1, num_reps):
                accept_prop = torch.exp((scales[i] - scales[i-1])*(energy[i] - energy[i-1]))
                accept_flag = torch.rand(1) < accept_prop
                if accept_flag.item():
                    tmp = x[i].clone()
                    x[i] = x[i-1]
                    x[i-1] = tmp

                    tmp = energy[i].clone()
                    energy[i] = energy[i-1]
                    energy[i-1] = tmp

            if k >= 10000:
                x_record.append(x.clone().numpy())

    x_record = np.array(x_record)
    x_samples = x_record[:,-1,:]
    return x_samples   
 
if __name__ == "__main__":
    x1_min, x1_max = -1.5, 1.0
    x2_min, x2_max = -0.5, 2.0
    fig, axes = plt.subplots()
    scale = 0.05

    ## draw samples from the Müller potential using temperature replica exchange
    ## Monte Carlo sampling
    ############################################################################

    num_reps = 10 # number of replicas
    num_steps = 110000
    x_samples = Monte_Carlo_sampling(x1_min, x1_max, x2_min, x2_max, scale, num_reps, num_steps)

    #### plot samples
    fig = plt.figure()
    fig.clf()
    plt.plot(x_samples[:, 0], x_samples[:, 1], '.', alpha = 0.5)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel(r"$x_1$", fontsize=24)
    plt.ylabel(r"$x_2$", fontsize=24)
    axes.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("/data/home/yfsun/python_code/Muller_Brown_pot_sampling/mpsampling.png")
    plt.show()

    ####