import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 6))
    model_names = ['MF', 'MF+bias', 'Neural CF', 'Neural CF+bias']
    GRID_ORDER, PLOT_ORDER, SCATTER_ORDER = 1, 2, 3
    plt.subplot(2, 2, 1)
    plt.xticks([i for i in range(0, 4)], model_names)
    plt.ylabel('Precision@10')
    plt.xlabel('Models')

    plt.ylim(0.168, 0.17)
    plt.grid(zorder=GRID_ORDER)
    plt.plot([i for i in range(0, 4)], [0.1690, 0.1690, 0.1689, 0.1690], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, 4)], [0.1690, 0.1690, 0.1689, 0.1690], color='red', zorder=SCATTER_ORDER)

    plt.subplot(2, 2, 2)
    plt.xticks([i for i in range(0, 4)], model_names)
    plt.ylabel('Recall@10')
    plt.xlabel('Models')

    plt.ylim(0.785, 0.787)
    plt.grid(zorder=GRID_ORDER)
    plt.plot([i for i in range(0, 4)], [0.7858, 0.7857, 0.7857, 0.7857], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, 4)], [0.7858, 0.7857, 0.7857, 0.7857], color='red', zorder=SCATTER_ORDER)

    plt.subplot(2, 2, 3)
    plt.xticks([i for i in range(0, 4)], model_names)
    plt.ylabel('NDCG@10')
    plt.xlabel('Models')

    plt.ylim(0.771, 0.773)
    plt.grid(zorder=GRID_ORDER)
    plt.plot([i for i in range(0, 4)], [0.7721, 0.7716, 0.7718, 0.7719], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, 4)], [0.7721, 0.7716, 0.7718, 0.7719], color='red', zorder=SCATTER_ORDER)

    plt.subplot(2, 2, 4)
    plt.xticks([i for i in range(0, 4)], model_names)
    plt.ylabel('RMSE')
    plt.xlabel('Models')

    plt.grid(zorder=GRID_ORDER)
    plt.plot([i for i in range(0, 4)], [7.7549, 6.3596, 1.8417, 1.8413], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, 4)], [7.7549, 6.3596, 1.8417, 1.8413], color='red', zorder=SCATTER_ORDER)

    plt.tight_layout()
    plt.savefig("assets/mf_results/mf_results.jpg", dpi=500)

if __name__ == '__main__':
    main()