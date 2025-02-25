"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import argparse
import logging
import os
from typing import Dict, Tuple

import numpy as np
import paddle
from paddle.io import DataLoader

from ...viz.viz_model import Viz
from ..embedding_model import EmbeddingTrainingHead

logger = logging.getLogger(__name__)

Optimizer = paddle.optimizer.Optimizer
Scheduler = paddle.optimizer.lr.LRScheduler


def set_seed(seed: int) -> None:
    """Set random seed

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    paddle.seed(seed)


class EmbeddingTrainer:
    """Trainer for Koopman embedding model

    Args:
        model (EmbeddingTrainingHead): Embedding training model
        args (TrainingArguments): Training arguments
        optimizers (Tuple[Optimizer, Scheduler]): Tuple of Pytorch optimizer and lr scheduler.
        viz (Viz, optional): Visualization class. Defaults to None.
    """

    def __init__(
        self,
        model: EmbeddingTrainingHead,
        args: argparse.ArgumentParser,
        optimizers: Tuple[Optimizer, Scheduler],
        viz: Viz = None,
    ) -> None:
        """Constructor"""
        self.model = model.to(args.device)
        self.args = args
        self.optimizers = optimizers
        self.viz = viz

        # Load pre-trained state dictionaries if necessary
        if self.args.epoch_start > 0:
            logger.info(
                "Attempting to load optimizer, model and scheduler from epoch: {:d}".format(
                    self.args.epoch_start
                )
            )

            optimizer_path = os.path.join(
                self.args.ckpt_dir, "optimizer{:d}.pdopt".format(self.args.epoch_start)
            )
            if os.path.isfile(optimizer_path):
                optimizer_dict = paddle.load(
                    optimizer_path, map_location=lambda storage, loc: storage
                )
                self.optimizers[0].set_state_dict(optimizer_dict)

            schedular_path = os.path.join(
                self.args.ckpt_dir, "scheduler{:d}.pdopt".format(self.args.epoch_start)
            )
            if os.path.isfile(schedular_path):
                schedular_dict = paddle.load(
                    schedular_path, map_location=lambda storage, loc: storage
                )
                self.optimizers[1].set_state_dict(schedular_dict)

            self.model.load_model(self.args.ckpt_dir, epoch=self.args.epoch_start)

        set_seed(self.args.seed)

    def train(self, training_loader: DataLoader, eval_dataloader: DataLoader) -> None:
        """Training loop for the embedding model

        Args:
            training_loader (DataLoader): Training dataloader
            eval_dataloader (DataLoader): Evaluation dataloader
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]
        # Loop over epochs
        for epoch in range(self.args.epoch_start + 1, self.args.epochs + 1):

            loss_total = 0.0
            loss_reconstruct = 0.0
            self.model.clear_gradients()
            for mbidx, inputs in enumerate(training_loader):

                loss0, loss_reconstruct0 = self.model(**inputs)
                loss0 = loss0.sum()

                loss_reconstruct = loss_reconstruct + loss_reconstruct0.sum()
                loss_total = loss_total + loss0.detach()
                # Backwards!
                loss0.backward()
                optimizer.step()
                optimizer.clear_grad()

                if mbidx + 1 % 10 == 0:
                    logger.info(
                        "Epoch {:d}: Completed mini-batch {}/{}.".format(
                            epoch, mbidx + 1, len(training_loader)
                        )
                    )

            # Progress learning rate scheduler
            lr_scheduler.step()
            cur_lr = optimizer.get_lr()
            logger.info(
                "Epoch {:d}: Training loss {:.03f}, Lr {:.05f}".format(
                    epoch, loss_total.numpy()[0], cur_lr
                )
            )

            # Evaluate current model
            if epoch % 5 == 0 or epoch == 1:
                output = self.evaluate(eval_dataloader, epoch=epoch)
                logger.info(
                    "Epoch {:d}: Test loss: {:.02f}".format(
                        epoch, output["test_error"].numpy()[0]
                    )
                )

            # Save model checkpoint
            if epoch % self.args.save_steps == 0:
                logger.info("Checkpointing model, optimizer and scheduler.")
                # Save model checkpoint
                self.model.save_model(self.args.ckpt_dir, epoch=epoch)
                paddle.save(
                    optimizer.state_dict(),
                    os.path.join(
                        self.args.ckpt_dir, "optimizer{:d}.pdopt".format(epoch)
                    ),
                )
                paddle.save(
                    lr_scheduler.state_dict(),
                    os.path.join(
                        self.args.ckpt_dir, "scheduler{:d}.pdopt".format(epoch)
                    ),
                )

    @paddle.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Run evaluation, plot prediction and return metrics.

        Args:
            eval_dataset (Dataset): Evaluation dataloader
            epoch (int, optional): Current epoch, used for naming figures. Defaults to 0.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        test_loss = 0
        for mbidx, inputs in enumerate(eval_dataloader):

            loss, state_pred, state_target = self.model.evaluate(**inputs)
            test_loss = test_loss + loss

            if self.viz is not None and mbidx == 0:
                self.viz.plotEmbeddingPrediction(state_pred, state_target, epoch=epoch)

            return {"test_error": test_loss / len(eval_dataloader)}
