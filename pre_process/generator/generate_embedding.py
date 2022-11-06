import torch
import numpy as np

from torchvision.models import resnet18

from common.utils import plt_show

NUM_CLASSES = 1020
# CHECKPOINT_PATH = 'E://Lec//classifier_of_generator//style//model_epoch_latest.pth'
# CHECKPOINT_PATH = '../../output/generator_style_classification_resnet18/20210619.124834/model_epoch_12.pth'
# CHECKPOINT_PATH = '../../output/generator_style_classification_resnet18/20210619.124834/model_epoch_40.pth'
CHECKPOINT_PATH = '../../output/generator_style_classification_resnet18/20210622.105206/model_epoch_30.pth'
# EMBEDDING_SAVE_PATH = '../../resources/style_embedding.txt'
# EMBEDDING_SAVE_PATH = '../../resources/20210619.124834.model_epoch_12.style_embedding.txt'
# EMBEDDING_SAVE_PATH = '../../resources/20210619.124834.model_epoch_40.style_embedding.txt'
EMBEDDING_SAVE_PATH = '../../resources/20210622.105206.model_epoch_30.style_embedding.txt'


@torch.no_grad()
def gen_embedding():
    """Get the average style embedding of every style by retrieving the weights in the fc layer
    """
    model = resnet18(pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    model.eval()

    # the weights in the fc layer is consider as the center for all samples from a specific writer
    # thus, we can set it as our style specific token e.g., embedding
    embedding = model.fc.weight
    print("===> embedding.shape: ", embedding.shape)
    embedding = embedding.cpu().numpy()
    plt_show(embedding)

    with open(EMBEDDING_SAVE_PATH, 'w') as f:
        np.savetxt(f, embedding, fmt='%.6f')


if __name__ == '__main__':
    gen_embedding()
