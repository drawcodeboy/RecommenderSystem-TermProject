import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.utils import shuffle

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

        self.user_int_map, self.item_int_map = None, None
        self._load()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]

        user_id = row['User-ID']
        item_id = row['ISBN']
        rating = row['Book-Rating'].astype(int)

        user_id_idx = self.user_int_map[user_id]
        item_id_idx = self.item_int_map[item_id]

        user_id_idx = torch.tensor(user_id_idx, dtype=torch.int64)
        item_id_idx = torch.tensor(item_id_idx, dtype=torch.int64)

        rating = torch.tensor(rating, dtype=torch.float32)
        return user_id_idx, item_id_idx, rating

    def _load(self):
        # (1) User id, Item id -> 정수 매핑 (nn.Embedding()에 넣게)
        user_set = set(self.ratings['User-ID'].to_list())
        item_set = set(self.ratings['ISBN'].to_list())
        print(len(user_set), len(item_set))

        self.user_int_map = {v: i for i, v in enumerate(user_set)}
        self.item_int_map = {v: i for i, v in enumerate(item_set)}

        # (2) Ratings가 0인 row 제거 (안 봤다는 거임)
        self.ratings = self.ratings[self.ratings['Book-Rating'] != 0]

        # (3) train, test split
        self.ratings = shuffle(self.ratings, random_state=42)
        cutoff = int(self.train_size * len(self.ratings))

        if self.mode == 'train':
            self.ratings = self.ratings.iloc[:cutoff]
        else:
            self.ratings = self.ratings.iloc[cutoff:]

    @classmethod
    def from_config(cls, cfg):
        return cls(mode=cfg.get('mode'),
                   train_size=cfg.get('train_size'),
                   data_path=cfg.get('data_path'))


if __name__ == '__main__':
    ds = BookDataset(mode='train'); print(len(ds))
    ds = BookDataset(mode='test'); print(len(ds))

    x = ds[0]
    for temp in x:
        print(temp.shape)