"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import os
import logging
import paddle
import paddle.nn as nn
from abc import abstractmethod
from trphysx.config.configuration_phys import PhysConfig

logger = logging.getLogger(__name__)


class EmbeddingModel(nn.Layer):
    """Parent class for embedding models that handle the projection of
    the physical systems states into a vector representation

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    model_name: str = "embedding_model"

    # Init config
    def __init__(self, config: PhysConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def embed(self, x):
        raise NotImplementedError("embed function has not been properly overridden")

    @abstractmethod
    def recover(self, x):
        raise NotImplementedError("recover function has not been properly overridden")

    @property
    @abstractmethod
    def koopmanOperator(self):
        pass

    @property
    @abstractmethod
    def koopmanDiag(self):
        pass

    @property
    def input_dims(self):
        return self.config.state_dims

    @property
    def embed_dims(self):
        return self.config.n_embd

    @property
    def devices(self):
        """Get list of unique device(s) model exists on"""
        devices = []
        for param in self.parameters():
            if param.place not in devices:
                devices.append(param.place)
        for buffer in self.buffers():
            if buffer.place not in devices:
                devices.append(buffer.place)
        return devices

    def save_model(self, save_directory: str, epoch: int = 0) -> None:
        """Saves embedding model to the specified directory.

        Args:
            save_directory (str): Folder directory to save state dictionary to.
            epoch (int, optional): Epoch of current model for file name. Defaults to 0.

        Raises:
            FileNotFoundError: If provided path is a file
        """
        if os.path.isfile(save_directory):
            raise FileNotFoundError(
                "Provided path ({}) should be a directory, not a file".format(
                    save_directory
                )
            )

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(
            save_directory, "{}{:d}.pdparams".format(self.model_name, epoch)
        )
        # Save pytorch model to file
        paddle.save(self.state_dict(), output_model_file)

    def load_model(self, file_or_path_directory: str, epoch: int = 0) -> None:
        """Load a embedding model from the specified file or path

        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
            epoch (int, optional): Epoch of current model for file name, used if folder path is provided. Defaults to 0.

        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info(
                "Loading embedding model from file: {}".format(file_or_path_directory)
            )
            self.set_state_dict(paddle.load(file_or_path_directory))
        elif os.path.isdir(file_or_path_directory):
            file_path = os.path.join(
                file_or_path_directory, "{}{:d}.pdparams".format(self.model_name, epoch)
            )
            logger.info("Loading embedding model from file: {}".format(file_path))
            self.set_state_dict(paddle.load(file_path))
        else:
            raise FileNotFoundError(
                "Provided path or file ({}) does not exist".format(
                    file_or_path_directory
                )
            )


class EmbeddingTrainingHead(nn.Layer):
    """Parent class for training head for embedding models"""

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward function has not been properly overridden")

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("evaluate function has not been properly overridden")

    def save_model(self, *args, **kwargs):
        """
        Saves the embedding model
        """
        assert (
            self.embedding_model is not None
        ), "Must initialize embedding model before saving."

        self.embedding_model.save_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Load the embedding model"""
        assert (
            self.embedding_model is not None
        ), "Must initialize embedding model before loading."

        self.embedding_model.load_model(*args, **kwargs)
