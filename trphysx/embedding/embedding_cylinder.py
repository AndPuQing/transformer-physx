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
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
from trphysx.config.configuration_phys import PhysConfig

from .embedding_model import EmbeddingModel, EmbeddingTrainingHead

logger = logging.getLogger(__name__)
# Custom types
Tensor = paddle.Tensor
TensorTuple = Tuple[paddle.Tensor]
FloatTuple = Tuple[float]


class CylinderEmbedding(EmbeddingModel):
    """Embedding Koopman model for the 2D flow around a cylinder system

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    model_name = "embedding_cylinder"

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method"""
        super().__init__(config)

        X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))
        self.mask = paddle.to_tensor(np.sqrt(X**2 + Y**2) < 1, dtype=paddle.bool)

        # # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2D(
                4, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode="replicate"
            ),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            # 8, 32, 64
            nn.Conv2D(
                16,
                32,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16, 16, 32
            nn.Conv2D(
                32,
                64,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # 16, 8, 16
            nn.Conv2D(
                64,
                128,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            # 16, 4, 8
            nn.Conv2D(
                128,
                config.n_embd // 32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )

        self.observableNetFC = nn.Sequential(
            # nn.Linear(config.n_embd // 32 * 4 * 8, config.n_embd-1),
            nn.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon),
            # nn.BatchNorm1d(config.n_embd, epsilon=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop),
        )

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2D(
                config.n_embd // 32,
                128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            # 16, 8, 16
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2D(
                128,
                64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            # 16, 16, 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2D(
                64,
                32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            # 8, 32, 64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2D(
                32,
                16,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.ReLU(),
            # 16, 64, 128
            nn.Conv2D(
                16, 3, kernel_size=(3, 3), stride=1, padding=1, padding_mode="replicate"
            ),
        )
        # # Learned Koopman operator parameters
        self.obsdim = config.n_embd
        # # We parameterize the Koopman operator as a function of the viscosity
        self.kMatrixDiagNet = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.obsdim)
        )
        # # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 5):
            yidx.append(np.arange(i, self.obsdim))
            xidx.append(np.arange(0, self.obsdim - i))
        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype=paddle.int64)
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype=paddle.int64)

        # # The matrix here is a small NN since we need to make it dependent on the viscosity
        self.kMatrixUT = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.shape[0])
        )
        self.kMatrixLT = nn.Sequential(
            nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.shape[0])
        )
        # # Normalization occurs inside the model
        self.register_buffer("mu", paddle.to_tensor([0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("std", paddle.to_tensor([1.0, 1.0, 1.0, 1.0]))

    def forward(self, x: Tensor, visc: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (TensorTuple): Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3, H, W] Recovered feature tensor
        """
        # Concat viscosities as a feature map
        x = paddle.concat(
            [x, visc.unsqueeze(-1).unsqueeze(-1) * paddle.ones_like(x[:, :1])], axis=1
        )
        x = self._normalize(x)
        g0 = self.observableNet(x)
        g = self.observableNetFC(g0.reshape((g0.shape[0], -1)))
        # Decode
        out = self.recoveryNet(g.reshape(g0.shape))
        xhat = self._unnormalize(out)
        # Apply cylinder mask
        self.mask = paddle.to_tensor(self.mask, dtype=paddle.int64)
        mask = self.mask.reshape((1, 1, self.mask.shape[0], self.mask.shape[1]))
        mask0 = paddle.repeat_interleave(mask, xhat.shape[0], axis=0)
        mask0 = paddle.repeat_interleave(mask0, xhat.shape[1], axis=1) is True
        # mask0 = self.mask.repeat(xhat.shape[0], xhat.shape[1], 1, 1) is True
        xhat[mask0] = 0

        return g, xhat

    def embed(self, x: Tensor, visc: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        # Concat viscosities as a feature map
        x = paddle.concat(
            [x, visc.unsqueeze(-1).unsqueeze(-1) * paddle.ones_like(x[:, :1])], axis=1
        )
        x = self._normalize(x)

        g = self.observableNet(x)
        g = self.observableNetFC(g.reshape((g.shape[0], -1)))
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, 3, H, W] Physical feature tensor
        """
        x = self.recoveryNet(g.reshape((-1, self.obsdim // 32, 4, 8)))
        x = self._unnormalize(x)
        # Apply cylinder mask
        self.mask = paddle.to_tensor(self.mask, dtype=paddle.int64)
        mask = self.mask.reshape((1, 1, self.mask.shape[0], self.mask.shape[1]))
        mask = paddle.repeat_interleave(mask, x.shape[0], axis=0)
        mask0 = paddle.repeat_interleave(mask, x.shape[1], axis=1) == 1
        x[mask0] = 0
        return x

    def koopmanOperation(self, g: Tensor, visc: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = paddle.to_tensor(
            paddle.zeros((g.shape[0], self.obsdim, self.obsdim)), stop_gradient=False
        )
        # Populate the off diagonal terms
        kMatrix[:, self.xidx, self.yidx] = self.kMatrixUT(100 * visc)
        kMatrix[:, self.yidx, self.xidx] = self.kMatrixLT(100 * visc)

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])
        self.kMatrixDiag = self.kMatrixDiagNet(100 * visc)
        kMatrix[:, ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = paddle.bmm(kMatrix, g.unsqueeze(-1))
        self.kMatrix = kMatrix
        return gnext.squeeze(-1)  # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool = True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            Tensor: Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x: Tensor) -> Tensor:
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(
            0
        ).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x + self.mu[
            :3
        ].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag


class CylinderEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method"""
        super().__init__()
        self.embedding_model = CylinderEmbedding(config)

    def forward(self, states: Tensor, viscosity: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            viscosity (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        assert (
            states.shape[0] == viscosity.shape[0]
        ), "State variable and viscosity tensor should have the same batch dimensions."

        self.embedding_model.train()

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:, 0]  # Time-step
        viscosity = viscosity

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0, viscosity)
        loss = (1e1) * mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0, :]  # Next time-step
            _, xRec1 = self.embedding_model(xin0, viscosity)
            # Apply Koopman transform
            g1Pred = self.embedding_model.koopmanOperation(g1_old, viscosity)
            xgRec1 = self.embedding_model.recover(g1Pred)

            # Loss function
            loss = (
                loss
                + (1e1) * mseLoss(xgRec1, xin0)
                + (1e1) * mseLoss(xRec1, xin0)
                + (1e-2) * paddle.sum(paddle.pow(self.embedding_model.koopmanDiag, 2))
            )

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(
        self, states: Tensor, viscosity: Tensor
    ) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            viscosity (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:, 1:]
        xInput = states[:, :-1]
        yPred = paddle.zeros_like(yTarget)
        viscosity = viscosity

        # Test accuracy of one time-step
        for i in range(xInput.shape[1]):
            xInput0 = xInput[:, i]
            g0 = self.embedding_model.embed(xInput0, viscosity)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget
