import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms.functional as TF
from tqdm import tqdm

# pip install pretrainedmodels
import pretrainedmodels

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
        # bboxes_original = self.labels[video_id][split_id][img_id]['boxes']
        # frame_idx  = self.labels[video_id][split_id][img_id]['frame_idx']
        an = torch.Tensor(keypoints_original)[:,:2]
        vis = torch.LongTensor(keypoints_original)[:,2]
        # bboxes_original[3] *= height
        # bboxes_original[1] *= height
        # bboxes_original[0] *= width
        # bboxes_original[2] *= width
        keypts = torch.as_tensor(keypoints_original, dtype = torch.float32)
        keypts *= torch.Tensor([width, height, 1])
        # bboxes = torch.as_tensor(bboxes_original, dtype = torch.float32).view(1, 4)

        # target = {}
        # target['boxes'] = bboxes
        # target['labels'] = torch.as_tensor([1 for _ in bboxes],
        #                                    dtype = torch.int64)
        # target['image_id'] = torch.tensor([index])
        # target['area'] = (bboxes[:,3] - bboxes[:,1]) * (bboxes[:,2] - bboxes[:,0])
        # target['iscrowd'] = torch.zeros(len(bboxes), dtype = torch.int64)
        # target['keypoints'] = keypts

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
            return img, keypts, (video_id, split_id, img_id)

        return img, keypts

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


class KeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(KeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Traning intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')

        self.l0 = nn.Linear(2048, 24*3)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0

def mpjpe(pred, true):
    '''
    Assume both inputs are (24,2) numpy arrays
    Compute the mean of Euclidean distance for all joints 
    '''
    return sum(np.linalg.norm(pred - true, axis=1)) / 24

def pck(pred, true, thereshold=20):
    '''
    Compute the percentage of correctly idenfied joints
    Correcly identified = predicted value within true value +- thereshold (in pixels)
    '''
    count = 0
    for i in range(24):
        pred_joint = pred[i,:]
        true_joint = true[i,:]
        if abs(pred_joint[0] - true_joint[0]) <= thereshold and abs(pred_joint[1] - true_joint[1]) <= thereshold:
            count += 1
    return count / 24

def pcp(pred, true, thereshold=0.5):
    '''
    Percentage of correct parts
    thereshold = 0.5 * limb length by default
    If start point and end point are a both within thereshold, the set of joints are classified as correct
    '''
    bones = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9],
             [2,10], [10,11], [11,12], [6,13], [13,14], [14,15],
             [16,17], [17,18], [18,19], [12,17], [12,18],
             [20,21], [21,22], [22,23], [15,21], [15,22]]
    count = 0
    for bone in bones:
        start_idx, end_idx = bone
        true_start, true_end = true[start_idx, :], true[end_idx, :]
        pred_start, pred_end = pred[start_idx, :], pred[end_idx, :]
        true_limb_length = np.linalg.norm(true_end - true_start)
        start_joint_error = np.linalg.norm(pred_start - true_start)
        end_joint_error = np.linalg.norm(pred_end - true_end)
        if start_joint_error <= thereshold * true_limb_length and end_joint_error <= thereshold * true_limb_length:
            count += 1
    return count / len(bones)
    
    
lr = 1e-3
batch_size = 10
num_epochs = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_extension, imgs_dir = determine_image_format()
label_path = 'target.json'
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
train_dataset = SkiDataset(imgs_dir=imgs_dir, label_path=label_path, img_extension=img_extension,
                        img_size=(224,224), mode='train', normalize=False, in_pixels=True, return_info=False,
                        transform=transform)
val_dataset = SkiDataset(imgs_dir=imgs_dir, label_path=label_path, img_extension=img_extension,
                        img_size=(224,224), mode='val', normalize=False, in_pixels=True, return_info=False,
                        transform = transform)
test_dataset = SkiDataset(imgs_dir=imgs_dir, label_path=label_path, img_extension=img_extension,
                        img_size=(224,224), mode='test', normalize=False, in_pixels=True, return_info=False,
                        transform = transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


    
train_losses = []
val_losses = []
lrs = []

model = KeypointResNet50(pretrained = True, requires_grad = True)
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr,
                           weight_decay = 1e-7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1} / {num_epochs}')

    print('Start training...')
    model.train()
    train_loss = 0.0
    num_batches = int(len(train_dataset)/train_dataloader.batch_size)

    for i,  (images, keypoints) in tqdm(enumerate(train_dataloader), total = num_batches):
        images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
        keypoints = keypoints.view(keypoints.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    print('Validating...')
    model.eval()
    val_loss = 0.0
    num_batches = int(len(val_dataset) / val_dataloader.batch_size)
    with torch.no_grad():
        for i, (images, keypoints) in tqdm(enumerate(val_dataloader), total = num_batches):
            images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            val_loss += loss.item()
        
    val_loss /= len(val_dataloader)
   
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    curr_lr = optimizer.param_groups[0]['lr']
    lrs.append(curr_lr)
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Current learning rate: {curr_lr:.4f}')
    scheduler.step(val_loss)
 
epochs = range(1, num_epochs+1)
fig1, ax1 = plt.subplots()
ax1.plot(epochs, train_losses, 'g', label='Training loss')
ax1.plot(epochs, val_losses, 'b', label='validation loss')
ax1.set(xlabel='Epochs', ylabel='Loss',
       title='Training and Validation loss')
ax1.legend()

fig1.savefig("loss.png")

fig2, ax2 = plt.subplots()
ax2.plot(epochs, lrs, 'r', label='Learning Rate')
ax2.set(xlabel='Epochs', ylabel='Loss',
       title='Learning Rate')
fig2.savefig("learning_rate.png")

# test
criterion = nn.MSELoss()
model.eval()
test_loss = 0.0
with torch.no_grad():
    for i, (images, keypoints) in enumerate(test_dataloader):
        images = images.to(DEVICE)
        keypoints = keypoints.to(DEVICE)
        keypoints = keypoints.view(keypoints.size(0), -1)
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        test_loss += loss.item()   
    test_loss /= len(test_dataloader)
print(f'test loss: {test_loss:.4f}')

mpjpe_eval = 0.0
pcp_eval = 0.0
pck_eval = 0.0
for image, keypoint in test_dataloader:
  image = image.to(DEVICE)
  pred = model(image).view(1, 24, 2)
  pred = pred[0].cpu().detach().numpy()
  keypoint = keypoint[0].numpy()
  mpjpe_eval += mpjpe(pred, keypoint)
  pcp_eval += pcp(pred, keypoint)
  pck_eval += pcp(pred, keypoint)
mpjpe_eval /= len(test_dataloader)
pcp_eval /= len(test_dataloader)
pck_eval /= len(test_dataloader)
