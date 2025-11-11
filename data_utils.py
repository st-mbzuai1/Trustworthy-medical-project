
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

HAM_LABELS = ['akiec','bcc','bkl','df','mel','nv','vasc']
LABEL2IDX = {c:i for i,c in enumerate(HAM_LABELS)}
IDX2LABEL = {i:c for c,i in LABEL2IDX.items()}

class HamDataset(Dataset):
    def __init__(self, csv_path, split='train', img_size=256):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split']==split].reset_index(drop=True)
        if split=='train':
            self.tf = T.Compose([
                T.Resize((img_size,img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(0.1,0.1,0.1,0.05),
                T.ToTensor()
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size,img_size)),
                T.ToTensor()
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(row['image_path']).convert('RGB')
        x = self.tf(x)
        y = LABEL2IDX[row['label']]
        return x, y, row['image_path']

def make_loader(csv_path, split, img_size, batch, shuffle):
    ds = HamDataset(csv_path, split, img_size)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)
