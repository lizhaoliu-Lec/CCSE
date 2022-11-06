import torch

from detectron2.engine import DefaultPredictor
from detectron2.modeling import detector_postprocess


class DefaultPredictorWithProposal(DefaultPredictor):
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            # get the proposal result
            if hasattr(self.model, 'proposal_generator'):
                images = self.model.preprocess_image([inputs])
                features = self.model.backbone(images.tensor)
                proposals, _ = self.model.proposal_generator(images, features)
                predictions.update({
                    'proposals': detector_postprocess(proposals[0],
                                                      height,
                                                      width)
                })
            return predictions
