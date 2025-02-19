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
import paddle
import paddle.nn as nn
import numpy as np
from typing import Tuple
from .embedding_model import EmbeddingModel, EmbeddingTrainingHead
from trphysx.config.configuration_phys import PhysConfig

logger = logging.getLogger(__name__)
# Custom types
Tensor = paddle.Tensor
TensorTuple = Tuple[paddle.Tensor]
FloatTuple = Tuple[float]


class LorenzEmbedding(EmbeddingModel):
    """Embedding Koopman model for the Lorenz ODE system

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    model_name = "embedding_lorenz"

    def __init__(self, config: PhysConfig):
        """Constructor method"""
        super().__init__(config)
        hidden_states = int(abs(config.state_dims[0] - config.n_embd) / 2) + 1
        hidden_states = 500

        self.observableNet = nn.Sequential(
            nn.Linear(config.state_dims[0], hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.n_embd),
            nn.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop),
        )
        self.recoveryNet = nn.Sequential(
            nn.Linear(config.n_embd, hidden_states),
            nn.ReLU(),
            nn.Linear(hidden_states, config.state_dims[0]),
        )
        # Learned Koopman operator
        self.obsdim = config.n_embd
        self.kMatrixDiag = paddle.create_parameter(
            shape=[config.n_embd],
            dtype="float32",
            default_initializer=nn.initializer.Assign(np.linspace(1, 0, config.n_embd)),
        )
        self.add_parameter("kMatrixDiag", self.kMatrixDiag)

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 3):
            yidx.append(np.arange(i, config.n_embd))
            xidx.append(np.arange(0, config.n_embd - i))

        self.xidx = paddle.to_tensor(np.concatenate(xidx), dtype="int64")
        self.yidx = paddle.to_tensor(np.concatenate(yidx), dtype="int64")
        self.kMatrixUT = paddle.create_parameter(
            shape=[self.xidx.shape[0]],
            dtype="float32",
            default_initializer=nn.initializer.Assign(
                np.random.rand(self.xidx.shape[0])
            ),
        )
        self.add_parameter("kMatrixUT", self.kMatrixUT)

        # Normalization occurs inside the model
        self.register_buffer("mu", paddle.to_tensor([0.0, 0.0, 0.0]), persistable=True)
        self.register_buffer("std", paddle.to_tensor([1.0, 1.0, 1.0]), persistable=True)
        # logger.info("Number of embedding parameters: {}".format(super().num_parameters))

    def forward(self, x: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3] Input feature tensor

        Returns:
            TensorTuple: Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3] Recovered feature tensor
        """
        # Encode
        x = self._normalize(x)
        x = paddle.to_tensor(x, dtype="float32", stop_gradient=False)
        g = self.observableNet(x)
        # Decode
        out = self.recoveryNet(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3] Input feature tensor

        Returns:
            Tensor: [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        x = paddle.to_tensor(x, dtype="float32", stop_gradient=False)
        g = self.observableNet(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            Tensor: [B, 3] Physical feature tensor
        """
        out = self.recoveryNet(g)
        x = self._unnormalize(out)
        return x

    def koopmanOperation(self, g: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        kMatrix = paddle.to_tensor(paddle.zeros((self.obsdim, self.obsdim)))
        # Populate the off diagonal terms
        kMatrix[self.xidx, self.yidx] = self.kMatrixUT
        kMatrix[self.yidx, self.xidx] = -self.kMatrixUT

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[0])
        kMatrix[ind[0], ind[1]] = self.kMatrixDiag

        # Apply Koopman operation
        gnext = paddle.bmm(
            kMatrix.expand((g.shape[0], kMatrix.shape[0], kMatrix.shape[0])),
            g.unsqueeze(-1),
        )
        self.kMatrix = kMatrix
        return gnext.squeeze(-1)  # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool = True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            (Tensor): Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x):
        return (x - self.mu.unsqueeze(0)) / self.std.unsqueeze(0)

    def _unnormalize(self, x):
        return self.std.unsqueeze(0) * x + self.mu.unsqueeze(0)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag


class LorenzEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    def __init__(self, config: PhysConfig):
        """Constructor method"""
        super().__init__()
        self.embedding_model = LorenzEmbedding(config)

    def forward(self, states: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor

        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:, 0]  # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0)
        xin0 = paddle.to_tensor(xin0, dtype="float32")
        loss = (1e4) * mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0, :]
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)
            xin0 = paddle.to_tensor(xin0, dtype="float32")
            loss = (
                loss
                + mseLoss(xgRec1, xin0)
                + (1e4) * mseLoss(xRec1, xin0)
                + (1e-1)
                * paddle.sum(paddle.pow(self.embedding_model.koopmanOperator, 2))
            )

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.

        Args:
            states (Tensor): [B, T, 3] Time-series feature tensor

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:, 1:]
        xInput = states[:, :-1]
        yPred = paddle.zeros(yTarget.shape)

        # Test accuracy of one time-step
        for i in range(xInput.shape[1]):
            xInput0 = xInput[:, i]
            g0 = self.embedding_model.embed(xInput0)
            g0 = self.embedding_model.koopmanOperation(g0)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:, i] = yPred0.squeeze().detach()

        yTarget = paddle.to_tensor(yTarget, dtype="float32")
        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget
