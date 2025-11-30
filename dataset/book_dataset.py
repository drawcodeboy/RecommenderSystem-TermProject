import pandas as pd
from torch.utils.data import Dataset

class BookDataset(Dataset):
    def __init__(self,
                 mode='train',
                 train_size=0.75,
                 data_path="data/BookRecommendation/Ratings.csv"):
        super().__init__()

        self.data_path = data_path
        self.mode = mode
        self.train_size = train_size
        self.ratings = pd.read_csv(self.data_path)

        self._load()

    def __len__():

    def __getitem__(self, idx):
        

    def _load():

    @classmethod
    def from_config(cls, cfg):
        return cls(mode=cfg.get('mode'),
                   train_size=cfg.get('train_size'),
                   data_path=cfg.get('data_path'))