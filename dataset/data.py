import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  transforms
from PIL import Image
import logging
HANDLE = "CAD"
logger = logging.getLogger(HANDLE)

class NABirdsDataset(Dataset):        
    def __init__(self, data_list) -> None:
        self.transforms = transforms.Compose([
                        transforms.Resize((256, 256), Image.BILINEAR),
                        # transforms.RandomCrop((384, 384)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        # normalize
                ])
        self.data_list = data_list
        print(f"Dataset has {len(self.data_list)} images.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, label


def get_data_loaders(cfg):
    root = cfg.DATA.DATA_PATH
    annotations_file = os.path.join(root, "image_class_labels.txt")
    img_paths_file = os.path.join(root, "images.txt")
    train_test_file = os.path.join(root, "train_test_split.txt")

    to_check = [annotations_file, img_paths_file, train_test_file]
    for paht_ in to_check:
        if not os.path.exists(paht_):
            raise FileNotFoundError(f"{paht_} not found.")

    image_paths = pd.read_csv(os.path.join(root,'images.txt'),sep=' ',names=['img_id','filepath'])
    image_class_labels = pd.read_csv(os.path.join(root,'image_class_labels.txt'),sep=' ',names=['img_id','target'])
    label_list = list(set(image_class_labels['target']))
    label_list = sorted(label_list)
    label_map = {k: i for i, k in enumerate(label_list)}
    train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
    data = image_paths.merge(image_class_labels, on='img_id')
    data = data.merge(train_test_split, on='img_id')

    logger.info(f"Loading data from {root} ...")

    train_data = data[data.is_training_img == 1]
    test_data = data[data.is_training_img == 0]

    train_images_and_targets = []
    for index,row in train_data.iterrows():
        file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
        target = int(label_map[row['target']])
        train_images_and_targets.append([file_path,target])

    test_images_and_targets = []
    for index,row in test_data.iterrows():
        file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
        target = int(label_map[row['target']])
        test_images_and_targets.append([file_path,target])

    train_loader = DataLoader(NABirdsDataset(train_images_and_targets), batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS, persistent_workers=True)
    test_loader = DataLoader(NABirdsDataset(test_images_and_targets), batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS, persistent_workers=True)
    return train_loader, test_loader


# def find_images_and_targets_nabirds(root,dataset,istrain=False,aux_info=False):
#     root = os.path.join(root,'nabirds')
#     image_paths = pd.read_csv(os.path.join(root,'images.txt'),sep=' ',names=['img_id','filepath'])
#     image_class_labels = pd.read_csv(os.path.join(root,'image_class_labels.txt'),sep=' ',names=['img_id','target'])
#     label_list = list(set(image_class_labels['target']))
#     label_list = sorted(label_list)
#     label_map = {k: i for i, k in enumerate(label_list)}
#     train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
#     data = image_paths.merge(image_class_labels, on='img_id')
#     data = data.merge(train_test_split, on='img_id')
#     if istrain:
#         data = data[data.is_training_img == 1]
#     else:
#         data = data[data.is_training_img == 0]
#     images_and_targets = []
#     images_info = []
#     for index,row in data.iterrows():
#         file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
#         target = int(label_map[row['target']])
#         images_and_targets.append([file_path,target])
#     return images_and_targets,None,images_info
    