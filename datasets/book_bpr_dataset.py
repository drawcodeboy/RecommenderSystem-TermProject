import torch
from torch.utils.data import Dataset
import random
import pandas as pd
from sklearn.utils import shuffle

class BookBPRDataset(Dataset):
    def __init__(self,
                 mode='train',
                 train_size=0.75,
                 data_path="data/BookRecommendation/Ratings.csv",
                 num_negatives=1):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.train_size = train_size
        self.num_negatives = num_negatives

        self.ratings = pd.read_csv(self.data_path)
        self.user_int_map, self.item_int_map = None, None
        self.user_items_dict = None  # user별 positive item set
        self.all_items = None

        self._load()
        print("_load() done!")
        self.triplets = self._make_triplets()
        print("_make_triplets() done!")

    def _load(self):
        # 0 rating 제거
        self.ratings = self.ratings[self.ratings['Book-Rating'] != 0]

        # 유저, 아이템 정수 매핑
        user_set = set(self.ratings['User-ID'].to_list())
        item_set = set(self.ratings['ISBN'].to_list())
        self.user_int_map = {v: i for i, v in enumerate(user_set)}
        self.item_int_map = {v: i for i, v in enumerate(item_set)}

        # train/test split
        self.ratings = shuffle(self.ratings, random_state=42)
        cutoff = int(self.train_size * len(self.ratings))
        if self.mode == 'train':
            self.ratings = self.ratings.iloc[:cutoff]
        else:
            self.ratings = self.ratings.iloc[cutoff:]

        # user별 positive item set
        self.user_items_dict = self.ratings.groupby('User-ID')['ISBN'].apply(set).to_dict()
        self.all_items = set(self.ratings['ISBN'].unique())

    def _make_triplets(self):
        triplets = []
        for u, pos_items in self.user_items_dict.items():
            for i in pos_items:
                for _ in range(0, self.num_negatives):
                    j = random.choice(list(self.all_items - pos_items))
                    triplets.append((u, i, j))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        u, i, j = self.triplets[idx]

        u_idx = torch.tensor(self.user_int_map[u], dtype=torch.int64)
        i_idx = torch.tensor(self.item_int_map[i], dtype=torch.int64)
        j_idx = torch.tensor(self.item_int_map[j], dtype=torch.int64)

        return u_idx, i_idx, j_idx
    
    @classmethod
    def from_config(cls, cfg):
        return cls(mode=cfg.get('mode'),
                   train_size=cfg.get('train_size'),
                   data_path=cfg.get('data_path'),
                   num_negatives=cfg.get('num_negatives'))