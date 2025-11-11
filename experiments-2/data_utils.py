import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

HAM_LABELS = ['akiec','bcc','bkl','df','mel','nv','vasc']
LABEL2IDX = {c:i for i,c in enumerate(HAM_LABELS)}
IDX2LABEL = {i:c for c,i in LABEL2IDX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class HamDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split='train',
        img_size=256,
        normalize=None,          # None | "imagenet" | "custom"
        mean=None,               # used only if normalize == "custom"
        std=None,                # used only if normalize == "custom"
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        tfm = [T.Resize((img_size, img_size))]

        if split == 'train':
            tfm += [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            ]

        # always go to tensor first
        tfm += [T.ToTensor()]

        # optional normalization
        if normalize is not None:
            if normalize == "imagenet":
                tfm += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            elif normalize == "custom":
                if mean is None or std is None:
                    raise ValueError("For normalize='custom' you must supply mean and std tuples.")
                tfm += [T.Normalize(mean, std)]
            else:
                raise ValueError("normalize must be None, 'imagenet', or 'custom'.")

        self.tf = T.Compose(tfm)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(row['image_path']).convert('RGB')
        x = self.tf(x)
        y = LABEL2IDX[row['label']]
        return x, y, row['image_path']

def make_loader(csv_path, split, img_size, batch, shuffle,
                normalize=None, mean=None, std=None):
    ds = HamDataset(
        csv_path, split, img_size,
        normalize=normalize, mean=mean, std=std
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)
