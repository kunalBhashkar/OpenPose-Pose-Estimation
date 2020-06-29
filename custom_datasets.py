import torch
import pandas as pd
from PIL import Image


class YogaPoseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dir, transforms, pose_id_to_name, pose_name_to_id):
        super(YogaPoseDataset, self).__init__()
        self.csv_dir = csv_dir
        self.transforms = transforms
        
        self.df = pd.read_csv(csv_dir)
        self.len = len(self.df.index)
        self.pose_id_to_name = pose_id_to_name
        self.pose_name_to_id = pose_name_to_id
           
    def __getitem__(self, index):
        img = Image.open(self.df.loc[index]['file_name'])
        img = self.transforms(img)
        
        label = self.df.loc[index]['pose_id']
        return (img, label)
    
    def __len__(self):
        return self.len
    
