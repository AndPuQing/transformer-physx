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

import h5py

# import paddle
import paddle

from ..embedding.embedding_model import EmbeddingModel
from .dataset_phys import PhysicalDataset

logger = logging.getLogger(__name__)


class CylinderDataset(PhysicalDataset):
    """Dataset for 2D flow around a cylinder numerical example"""

    def embed_data(self, h5_file: h5py.File, embedder: EmbeddingModel) -> None:
        """Embeds cylinder flow data into a 1D vector representation for the transformer.

        Args:
            h5_file (h5py.File): HDF5 file object of raw data
            embedder (EmbeddingModel): Embedding neural network
        """
        # Iterate through stored time-series
        samples = 0
        embedder.eval()
        for key in h5_file.keys():
            ux = paddle.to_tensor(h5_file[key + "/ux"])
            uy = paddle.to_tensor(h5_file[key + "/uy"])
            p = paddle.to_tensor(h5_file[key + "/p"])
            data_series = paddle.stack([ux, uy, p], axis=1).to(embedder.devices[0])
            visc = (2.0 / float(key)) * paddle.ones(ux.shape[0], 1).to(
                embedder.devices[0]
            )

            with paddle.no_grad():
                embedded_series = embedder.embed(data_series, visc).cpu()

            # Stride over time-series
            for i in range(
                0, data_series.size(0) - self.block_size + 1, self.stride
            ):  # Truncate in block of block_size

                data_series0 = embedded_series[i : i + self.block_size]  # .repeat(1, 4)
                self.examples.append(data_series0)

                if self.eval:
                    self.states.append(data_series[i : i + self.block_size].cpu())

            samples = samples + 1
            if (
                self.ndata > 0 and samples >= self.ndata
            ):  # If we have enough time-series samples break loop
                break

        logger.info(
            "Collected {:d} time-series from hdf5 file. Total of {:d} time-series.".format(
                samples, len(self.examples)
            )
        )
