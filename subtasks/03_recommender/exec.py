import torch
from torch import nn
import pandas as pd
import numpy as np

import time, sys, os
sys.path.append(os.getcwd())

from models import load_model

ratings = pd.read_csv("data/BookRecommendation/Ratings.csv")
books = pd.read_csv("data/BookRecommendation/Books.csv", low_memory=False)

def get_idx_from_user_id(user_id):
    user_set = set(ratings['User-ID'].to_list())
    user_int_map = {v: i for i, v in enumerate(user_set)}

    user_idx = user_int_map[user_id]

    unrated = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] == 0)]
    rated = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] != 0)]

    return user_idx, unrated, rated

def print_rated_infos(rated, rank=5):
    rated = rated.sort_values(by='Book-Rating', ascending=False)
    rated_books = books[books['ISBN'].isin(rated['ISBN'].to_list())]

    titles = rated_books['Book-Title'].to_list()
    authors = rated_books['Book-Author'].to_list()
    
    print(f"User는 이런 책들을 좋아했어요! ({rank}등까지 보여줄게요!)")
    for i, (title, author) in enumerate(zip(titles, authors), start=1):
        if len(title) >= 50:
            title = title[:47] + '...'
        print(f"\t{i}. [제목]: {title:50s} - [저자]: {author:30s}")
        if i == rank: break
    print(f"위 정보들을 바탕으로 책 추천을 시작합니다...")

def print_recommender(unrated, indices, rank=5):
    indices = indices.cpu().detach().numpy()

    rank = min(len(indices), rank)

    inv = np.empty_like(indices)
    inv[indices] = np.arange(len(indices))
    
    isbn_li = []
    for i in range(rank):
        isbn = books.iloc[inv[i]]['ISBN']
        isbn_li.append(isbn)

    rec_books = books[books['ISBN'].isin(isbn_li)]

    titles = rec_books['Book-Title'].to_list()
    authors = rec_books['Book-Author'].to_list()
    
    print(f"[AI 추천 결과] >>>")
    print(f"User는 이런 책들을 좋아할 거 같아요! ({rank}등까지 보여줄게요!)")
    for i, (title, author) in enumerate(zip(titles, authors), start=1):
        if len(title) >= 50:
            title = title[:47] + '...'
        print(f"\t{i}. [제목]: {title:50s} - [저자]: {author:30s}")
        if i == rank: break

def get_batch(user_idx, unrated):
    item_set = set(ratings['ISBN'].to_list())
    item_int_map = {v: i for i, v in enumerate(item_set)}

    isbn_li = unrated['ISBN'].to_list()
    batch_li = []
    
    for isbn in isbn_li:
        item_idx = item_int_map[isbn]
        batch_li.append(item_idx)
    
    b_items = torch.tensor(batch_li, dtype=torch.int64)
    b_users = torch.tensor(len(batch_li) * [user_idx], dtype=torch.int64)

    return b_users, b_items

def main():
    user_id = int(input("User의 번호를 입력하세요 (데이터셋 내 Users.csv / Ratings.csv 참고): ")) # 238419

    user_idx, unrated, rated = get_idx_from_user_id(user_id)
    print_rated_infos(rated)

    b_users, b_items = get_batch(user_idx, unrated)

    # Device Setting
    device = None
    if torch.cuda.is_available():
        device = 'cuda:0'
    else: 
        device = 'cpu'
    print(f"AI 사용 device: {device}")
    
    # Load Model
    model_cfg = {
        "name": "NeuralCF",
        "num_users": 105283, # dataset script로 확인
        "num_items": 340556, # dataset script로 확인
        "latent_dim": 200,
        "use_bias": True
    }
    model = load_model(model_cfg).to(device)
    ckpt = torch.load(os.path.join('saved/weights/ncf_bias.book.epochs_100.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    preds = []
    with torch.no_grad():
        model.eval()
        for i in range(b_users.shape[0]):
            b_user = b_users[i].unsqueeze(dim=0)
            b_item = b_items[i].unsqueeze(dim=0)
            pred = model(b_user.to(device), b_item.to(device))
            preds.append(pred)
            print(f"\rAI 예측 중...", end="")
    preds = torch.tensor(preds)
    _, indices = torch.sort(preds, descending=True)
    print()

    print_recommender(unrated, indices, rank=5)

if __name__ == '__main__':
    main()