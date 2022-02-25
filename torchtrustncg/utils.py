# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch


def rosenbrock(tensor, alpha=1.0, beta=100):
    x, y = tensor[..., 0], tensor[..., 1]
    return (alpha - x) ** 2 + beta * (y - x ** 2) ** 2


def branin(tensor, **kwargs):
    x, y = tensor[..., 0], tensor[..., 1]
    loss = ((y - 0.129 * x ** 2 + 1.6 * x - 6) ** 2 + 6.07 * torch.cos(x) + 10)
    return loss
