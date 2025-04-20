import torch
import json
import cv2
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SkiDataset(Dataset):
    '''2D alpine skiing dataset'''

    # BGR channel-wise mean and std
    all_mean = torch.Tensor([190.24553031, 176.98437134, 170.87045832]) / 255
    all_std  = torch.Tensor([ 36.57356531,  35.29007466,  36.28703238]) / 255
    train_mean = torch.Tensor([190.37117484, 176.86400202, 170.65409075]) / 255
    train_std  = torch.Tensor([ 36.56829177,  35.27981661,  36.19375109]) / 255

    # Corresponding joints
    joints = ['head', 'neck',
            'shoulder_right', 'elbow_right', 'hand_right', 'pole_basket_right',
            'shoulder_left', 'elbow_left', 'hand_left', 'pole_basket_left',
            'hip_right', 'knee_right', 'ankle_right',
            'hip_left', 'knee_left', 'ankle_left',
            'ski_tip_right', 'toes_right', 'heel_right', 'ski_tail_right',
            'ski_tip_left', 'toes_left', 'heel_left', 'ski_tail_left']

    # Bones for drawing examples
    bones = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9],
            [2,10], [10,11], [11,12], [6,13], [13,14], [14,15],
            [16,17], [17,18], [18,19], [12,17], [12,18],
            [20,21], [21,22], [22,23], [15,21], [15,22]]

    # VideoIDs and SplitIDs used for validation
    val_splits = [('5UHRvqx1iuQ', '0'), ('5UHRvqx1iuQ', '1'),
                ('oKQFABiOTw8', '0'), ('oKQFABiOTw8', '1'), ('oKQFABiOTw8', '2'),
                ('qxfgw1Kd98A', '0'), ('qxfgw1Kd98A', '1'),
                ('uLW74013Wp0', '0'), ('uLW74013Wp0', '1'),
                ('zW1bF2PsB0M', '0'), ('zW1bF2PsB0M', '1'),
                ('he3w2n9WvrI', '0'), ('he3w2n9WvrI', '1'),
                ('NrWcP1s3QC0', '0'), ('NrWcP1s3QC0', '1'), ('NrWcP1s3QC0', '2')]

    # VideoIDS and SplitIDs used for testing
    test_splits = [('he3w2n9WvrI', '3'), ('he3w2n9WvrI', '4'), ('he3w2n9WvrI', '5'),
                ('5UHRvqx1iuQ', '3'), ('5UHRvqx1iuQ', '4'),
                ('oKQFABiOTw8', '3'), ('oKQFABiOTw8', '4'), ('oKQFABiOTw8', '5'),
                ('qxfgw1Kd98A', '2'), ('qxfgw1Kd98A', '4'),
                ]

    def __init__(self, imgs_dir, label_path, img_extension='png', mode='all', img_size=(1920,1080),
                normalize=True, in_pixels=True, return_info=False, transform=None):
        '''
        Create a Ski2DPose dataset loading train or validation images.

        Args:
            :imgs_dir: Root directory where images are saved
            :label_path: Path to label JSON file
            :img_extension: Image format extension depending on downloaded version. One of {'png', 'jpg', 'webp'}
            :mode: Specify which partition to load. One of {'train', 'val', 'all'}
            :img_size: Size of images to return
            :normalize: Set to True to normalize images
            :in_pixels: Set to True to scale annotations to pixels
            :return_info: Set to True to include image names when getting items
        '''
        self.imgs_dir = imgs_dir
        self.img_extension = img_extension
        self.mode = mode
        self.img_size = img_size
        self.normalize = normalize
        self.in_pixels = in_pixels
        self.return_info = return_info
        self.transform = transform # Added

        assert mode in ['train', 'val', 'test', 'all'], 'Please select a valid mode.'
        self.mean = self.all_mean if self.mode == 'all' else self.train_mean
        self.std = self.all_std if self.mode == 'all' else self.train_std

        # Load annotations
        with open(label_path) as f:
            self.labels = json.load(f)

        # Check if all images exist and index them
        self.index_list = []
        self.img_ids = []
        for video_id, all_splits in self.labels.items():
            for split_id, split in all_splits.items():
                for img_id, img_labels in split.items():
                    img_path = os.path.join(imgs_dir, video_id, split_id, '{}.{}'.format(img_id, img_extension))
                    if os.path.exists(img_path):
                        if ((mode == 'all') or
                            (mode == 'train' and (video_id, split_id) not in self.val_splits+self.test_splits) or
                            (mode == 'val' and (video_id, split_id) in self.val_splits) or
                            (mode == 'test' and (video_id, split_id) in self.test_splits)):
                            self.index_list.append((video_id, split_id, img_id))
                            self.img_ids.append(img_id)
                    else:
                        print('Did not find image {}/{}/{}.{}'.format(video_id, split_id, img_id, img_extension))

    def __len__(self):
        '''Returns the number of samples in the dataset'''
        return len(self.index_list)

    def __getitem__(self, index):
        '''
        Returns image tensor (H x W x C) in BGR format with values between 0 and 255,
        annotation tensor (J x 2) and visibility flag tensor (J).

        If 'normalize' flag was set to True, returned image will be normalized
        according to mean and std above.

        If 'return_info' flag was set to True, returns (video_id, split_id, img_id, frame_idx)
        in addition.

        Args:
            :index: Index of data sample to load
        '''
        video_id, split_id, img_id = self.index_list[index]
        # Load image
        img_path = os.path.join(self.imgs_dir, video_id, split_id, '{}.{}'.format(img_id, self.img_extension))
        img = cv2.imread(img_path) # (H x W x C) BGR
        height = img.shape[0]
        width = img.shape[1]
        if self.img_size is not None:
            img = cv2.resize(img, self.img_size)
            width, height = self.img_size
        img = torch.from_numpy(img)

        # Added: reorder dimensions
        img = img.permute(2, 0, 1)

        # Load annotations
        keypoints_original = self.labels[video_id][split_id][img_id]['keypoints']
        bboxes_original = self.labels[video_id][split_id][img_id]['boxes']
        # frame_idx  = self.labels[video_id][split_id][img_id]['frame_idx']
        an = torch.Tensor(keypoints_original)[:,:2]
        vis = torch.LongTensor(keypoints_original)[:,2]
        bboxes_original[3] *= height
        bboxes_original[1] *= height
        bboxes_original[0] *= width
        bboxes_original[2] *= width
        keypts = torch.as_tensor(keypoints_original, dtype = torch.float32)
        keypts *= torch.Tensor([width, height, 1])
        bboxes = torch.as_tensor(bboxes_original, dtype = torch.float32).view(1, 4)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = torch.as_tensor([1 for _ in bboxes],
                                        dtype = torch.int64)
        target['image_id'] = torch.tensor([index])
        target['area'] = (bboxes[:,3] - bboxes[:,1]) * (bboxes[:,2] - bboxes[:,0])
        target['iscrowd'] = torch.zeros(len(bboxes), dtype = torch.int64)
        target['keypoints'] = keypts

        if self.normalize:
            img = ((img / 255) - self.mean) / self.std
        if self.in_pixels:
            an *= torch.Tensor([width, height])


        # Added
        if self.transform:
            img = self.transform(img)  # Apply the transform to the image
        else:
            img = TF.to_tensor(img)

        if self.return_info:
            # return img, an, vis, (video_id, split_id, img_id, frame_idx)
            return img, target, (video_id, split_id, img_id)

        return img, target

def determine_image_format():
    """
    @return (image_format_name, image_directory)
    """
    formats = ['png', 'webp', 'jpg']

    for img_format in formats:
        img_dir = Path(f'data/Images_{img_format}')
        if img_dir.is_dir():
            print(f'Found image directory {img_dir}, using {img_format} format.')
            return img_format, img_dir

    raise FileNotFoundError('Image directory not found, please ensure one of the following directories exists: ' + \
                            ', '.join(f'Images_{ext}' for ext in formats))