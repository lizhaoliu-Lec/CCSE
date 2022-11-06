from dataset.chinese_handwritting.hwdb import HWDB
from common.utils import join

if __name__ == '__main__':
    def run_preprocess():
        # root = 'E:/Datasets/HWD'
        root = '/mnt/cephfs/mixed/dataset/HWDB'
        # versions = ['1.0', '1.1', '1.2']
        # versions = ['1.0']
        versions = ['1.0', '1.2']
        # split = 'train'
        # # HWDB.pre_process_into_image_folder(root=root, versions=versions, split='train', chinese_only=True, rebuild=True)
        # image_save_dir = join(root, '-'.join(versions), 'image', 'new_test_v1')
        # HWDB.pre_process_into_image_folder(root=root, versions=versions,
        #                                    split='test', chinese_only=True, rebuild=True,
        #                                    image_save_dir=image_save_dir)
        HWDB.pre_process_into_image_folder(root=root, versions=versions,
                                           split='train', chinese_only=True, rebuild=True)
        HWDB.pre_process_into_image_folder(root=root, versions=versions,
                                           split='test', chinese_only=True, rebuild=True)


    run_preprocess()
