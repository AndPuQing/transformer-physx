"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
from dataclasses import dataclass
from typing import Dict, List

import paddle

Tensor = paddle.Tensor


@dataclass
class DataCollator:
    """
    Data collator used for training datasets.
    Combines examples in a minibatch into one tensor.

    Args:
        examples (List[Dict[str, Tensor]]): List of training examples. An example
            should be a dictionary of tensors from the dataset.

        Returns:
            Dict[str, Tensor]: Minibatch dictionary of combined example data tensors
    """

    # Default collator
    def __call__(self, examples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        mini_batch = {}
        for key in examples[0].keys():
            mini_batch[key] = self._tensorize_batch(
                [example[key] for example in examples]
            )

        return mini_batch

    def _tensorize_batch(self, examples: List[Tensor]) -> Tensor:
        if not paddle.is_tensor(examples[0]):
            return examples

        length_of_first = examples[0].shape[0]
        are_tensors_same_length = all(x.shape[0] == length_of_first for x in examples)

        if are_tensors_same_length:
            return paddle.stack(examples, axis=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return
