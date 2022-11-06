import torch
import numpy as np


class AsTensor:
    """Perform as Tensor by not changing pixel value from [0, 255] to [0.0, 1.0]
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        # print("===> type(pic): ", type(pic))
        x = torch.Tensor(np.array(pic),).permute(2, 0, 1)
        # print("===> x.type(): ", type(x), x.dtype)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
