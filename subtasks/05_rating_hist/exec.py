import matplotlib.pyplot as plt
import pandas as pd

def main():
    ratings = pd.read_csv("data/BookRecommendation/Ratings.csv")
    ratings = ratings[ratings['Book-Rating'] > 0]

    # print(len(ratings))
    plt.figure(figsize=(8, 4))
    plt.grid(zorder=1)

    plt.xticks([i for i in range(1, 10+1)])
    plt.hist(ratings['Book-Rating'].to_list(), bins=9, zorder=2)
    plt.xlabel('Ratings')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("assets/rating_hist/rating_hist.jpg", dpi=500)

if __name__ == '__main__':
    main()