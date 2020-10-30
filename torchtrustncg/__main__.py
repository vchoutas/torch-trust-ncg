# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de

import sys
import argparse
import torch
import numpy as np

from .utils import rosenbrock, branin
from . import TrustRegionNewtonCG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='rosenbrock',
                        choices=['rosenbrock', 'branin'],
                        help='Test function to optimize'
                        )
    parser.add_argument('--num-iters', dest='num_iters', default=100,
                        type=str,
                        help='Number of iterations to use')
    parser.add_argument('--gtol', default=1e-4, type=float,
                        help='Gradient tolerance')
    parser.add_argument('--init-point', nargs=2, default=[2.5, 2.5],
                        type=float, dest='init_point',
                        help='Point to initialize the optimization')

    args = parser.parse_args()
    function = args.function
    num_iters = args.num_iters
    gtol = args.gtol
    init_point = args.init_point

    if function == 'rosenbrock':
        loss_func = rosenbrock
    elif function == 'branin':
        loss_func = branin
    else:
        print(f'Unknown loss function {function}')
        sys.exit(-1)

    variable = torch.empty([1, 2], requires_grad=True)
    with torch.no_grad():
        variable[0, 0] = init_point[0]
        variable[0, 1] = init_point[1]

    optimizer = TrustRegionNewtonCG([variable])

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()
        loss = loss_func(variable)
        if backward:
            loss.backward(create_graph=True)
        return loss

    for n in range(num_iters):
        loss = optimizer.step(closure)

        np_var = variable.detach().cpu().numpy().squeeze()

        if torch.norm(variable.grad).item() < gtol:
            break
        if torch.norm(optimizer.param_step, dim=-1).lt(gtol).all():
            break

        print(
            f'[{n:04d}]: ' +
            f'Loss at ({variable[0, 0]:.4f}, {variable[0, 1]:.4f}) = ' +
            f'{loss.item():.4f}')
