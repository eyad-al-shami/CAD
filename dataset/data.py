import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import  transforms
from PIL import Image
import logging
HANDLE = "CAD"
logger = logging.getLogger(HANDLE)

class NABirdsDataset(Dataset):
    # def __init__(self, annotations_file, img_paths_file,  train_test_file, img_dir, train=True, target_transform=None):

    #     with open(annotations_file, "r", encoding="utf-8") as ids_classes_f, open(img_paths_file, "r", encoding="utf-8") as imgs_paths_f, open(train_test_file, "r", encoding="utf-8") as train_test_f:
    #       imgs_paths = imgs_paths_f.readlines()
    #       ids_classes = ids_classes_f.readlines()
    #       train_test = train_test_f.readlines()

    #       imgs_paths = [line.strip() for line in imgs_paths]
    #       ids_classes = [line.strip() for line in ids_classes]
    #       train_test = [line.strip() for line in train_test]

    #       imgs_paths = {line.split(" ")[0]: line.split(" ")[1] for line in imgs_paths}
    #       ids_classes = {line.split(" ")[0]: line.split(" ")[1] for line in ids_classes}
    #       train_test = {line.split(" ")[0]: int(line.split(" ")[1]) for line in train_test}

    #       for id, is_train_example in train_test.items():
    #         if (train and not is_train_example):
    #           del ids_classes[id]
    #           del imgs_paths[id]
    #         elif (not train and is_train_example):
    #           del ids_classes[id]
    #           del imgs_paths[id]

    #       self.imgs = []
    #       for id, label in ids_classes.items():
    #         self.imgs.append((id, label, imgs_paths[id]))

    #     self.img_dir = img_dir

    #     normalize = transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             )
    #     self.transforms = transforms.Compose([
    #                     transforms.Resize((512, 512), Image.BILINEAR),
    #                     # transforms.RandomCrop((384, 384)),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
    #                     transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
    #                     transforms.ToTensor(),
    #                     # normalize
    #             ])
        
    def __init__(self, data_list) -> None:
        self.transforms = transforms.Compose([
                        transforms.Resize((512, 512), Image.BILINEAR),
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
        # return len(self.imgs)
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
    img_dir = os.path.join(root, "images")

    to_check = [annotations_file, img_paths_file, train_test_file]
    for paht_ in to_check:
        if not os.path.exists(paht_):
            raise FileNotFoundError(f"{paht_} not found.")
        
    # train_dataloader = DataLoader(NABirdsDataset(annotations_file, img_paths_file, train_test_file, img_dir, train=True), batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    # test_dataloader = DataLoader(NABirdsDataset(annotations_file, img_paths_file, train_test_file, img_dir, train=False), batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    # return train_dataloader, test_dataloader

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

    images_and_targets = []
    for index,row in data.iterrows():
        file_path = os.path.join(os.path.join(root,'images'),row['filepath'])
        target = int(label_map[row['target']])
        images_and_targets.append([file_path,target])

    train_loader = DataLoader(NABirdsDataset(images_and_targets), batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS, persistent_workers=True)
    test_loader = DataLoader(NABirdsDataset(images_and_targets), batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS, persistent_workers=True)
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
    