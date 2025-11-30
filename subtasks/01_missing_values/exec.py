import pandas as pd

ratings = pd.read_csv("dataset/BookRecommendation/Ratings.csv")

print(f"Number of rows = {len(ratings)}, Number of columns = {len(ratings.columns)}")

print(f"User-item matrix (can't obtain it, because it demands a lot of memories.)")
print(f"Expected number of rows: {len(set(ratings['User-ID'].to_list()))}")
print(f"Expected number of columns: {len(set(ratings['ISBN'].to_list()))}")

user_item_matrix = None

try:
    user_item_matrix = ratings.pivot_table(
        index='User-ID',
        columns='ISBN',
        values='Book-Rating',
    )
    print(user_item_matrix.shape)

except:
    print("Memory Error...")