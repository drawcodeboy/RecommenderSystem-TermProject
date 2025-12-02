from .book_dataset import BookDataset
from .book_bpr_dataset import BookBPRDataset

def load_dataset(cfg):
    if cfg['name'] == 'book':
        return BookDataset.from_config(cfg)
    elif cfg['name'] == 'book_bpr':
        return BookBPRDataset.from_config(cfg)
    else:
        raise Exception(f"Check your cfg['name']: {cfg['name']}")