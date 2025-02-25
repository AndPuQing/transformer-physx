"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
import math
from typing import Callable

import paddle
import paddle.nn.functional as F
from paddle import nn

logger = logging.getLogger(__name__)

Tensor = paddle.Tensor


class Conv1D(nn.Layer):
    """1D-convolutional layer (eqv to FCN) as defined by Radford et al. for OpenAI GPT
    (and also used in GPT-2). Basically works like a linear layer but the weights are transposed.

    Note:
        Code adopted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
    """

    def __init__(self, nf: int, nx: int) -> None:
        """Constructor"""
        super().__init__()
        self.nf = nf
        # nn.init.normal_(w, std=0.02)
        self.weight = paddle.create_parameter(
            (nx, nf),
            dtype="float32",
            default_initializer=paddle.nn.initializer.Normal(std=0.02),
        )
        self.bias = paddle.create_parameter([nf], dtype="float32")
        self.add_parameter("weight", self.weight)
        self.add_parameter("bias", self.bias)
        # self.weight = nn.Parameter(w)
        # self.bias = nn.Parameter(paddle.zeros(nf))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [..., nx] input features

        Returns:
            Tensor: [..., nf] output features
        """
        size_out = list(x.shape[:-1]) + [self.nf]
        # size_out = x.shape[:-1] + (self.nf,)
        x = paddle.addmm(
            self.bias.reshape((1, -1)), x.reshape((-1, x.shape[-1])), self.weight
        )
        x = x.reshape(size_out)
        return x


def gelu_new(x: Tensor) -> Tensor:
    """Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT)."""
    return (
        0.5
        * x
        * (
            1.0
            + paddle.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))
            )
        )
    )


def gelu_fast(x):
    """Faster approximate form of GELU activation function"""
    return 0.5 * x * (1.0 + paddle.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x: Tensor) -> Tensor:
    """Mish activation function"""
    return x * paddle.tanh(paddle.nn.functional.softplus(x))


def linear_act(x: Tensor) -> Tensor:
    """Linear activate function"""
    return x


ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "tanh": paddle.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": paddle.nn.functional.sigmoid,
}


def get_activation(activation_string: str) -> Callable:
    """Gets a activation function

    Args:
        activation_string (str): Name of activate function

    Raises:
        KeyError: Not a valid activation function

    Returns:
        Callable: activate function
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            "function {} not found in ACT2FN mapping {}".format(
                activation_string, list(ACT2FN.keys())
            )
        )
