from .book_dataset import BookDataset

def load_dataset(cfg):
    if cfg['name'] == 'book':
        return BookDataset.from_config(cfg)
    else:
        raise Exception(f"Check your cfg['name']: {cfg['name']}")