import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 6))
    model_names = ['KNN', 'KNN+bias', 'MF', 'MF+bias', 'Neural CF', 'Neural CF+bias']
    GRID_ORDER, PLOT_ORDER, SCATTER_ORDER = 1, 2, 3
    ROTATION = 25
    plt.subplot(2, 2, 1)
    plt.xticks([i for i in range(0, len(model_names))], model_names, rotation=ROTATION)
    plt.ylabel('Precision')
    plt.xlabel('Models')

    plt.ylim(16.0, 29.0)
    plt.grid(zorder=GRID_ORDER)
    prec_top_10 = [0.1687, 0.1684, 0.1690, 0.1690, 0.1689, 0.1690]
    prec_top_5 = [0.2788, 0.2781, 0.2784, 0.2789, 0.2788, 0.2788]
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in prec_top_10], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in prec_top_5], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in prec_top_10], color='red', zorder=SCATTER_ORDER, label='Top@10')
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in prec_top_5], color='blue', zorder=SCATTER_ORDER, label='Top@5')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.xticks([i for i in range(0, len(model_names))], model_names, rotation=ROTATION)
    plt.ylabel('Recall')
    plt.xlabel('Models')

    plt.ylim(75.0, 79.0)
    plt.grid(zorder=GRID_ORDER)
    rec_top_10 = [0.7846, 0.7844, 0.7858, 0.7857, 0.7857, 0.7857]
    rec_top_5 = [0.7553, 0.7546, 0.7556, 0.7558, 0.7558, 0.7558]
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in rec_top_10], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in rec_top_5], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in rec_top_10], color='red', zorder=SCATTER_ORDER, label='Top@10')
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in rec_top_5], color='blue', zorder=SCATTER_ORDER, label='Top@5')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.xticks([i for i in range(0, len(model_names))], model_names, rotation=ROTATION)
    plt.ylabel('NDCG')
    plt.xlabel('Models')

    plt.ylim(76.3, 77.5)
    plt.grid(zorder=GRID_ORDER)
    ndcg_top_10 = [0.7720, 0.7707, 0.7721, 0.7716, 0.7718, 0.7719]
    ndcg_top_5 = [0.7670, 0.7653, 0.7657, 0.7660, 0.7661, 0.7661]
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in ndcg_top_10], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.plot([i for i in range(0, len(model_names))], [x * 100 for x in ndcg_top_5], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in ndcg_top_10], color='red', zorder=SCATTER_ORDER, label='Top@10')
    plt.scatter([i for i in range(0, len(model_names))], [x * 100 for x in ndcg_top_5], color='blue', zorder=SCATTER_ORDER, label='Top@5')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.xticks([i for i in range(0, len(model_names))], model_names, rotation=ROTATION)
    plt.ylabel('RMSE')
    plt.xlabel('Models')

    plt.grid(zorder=GRID_ORDER)
    plt.plot([i for i in range(0, len(model_names))], [1.8946, 1.8139, 7.7549, 6.3596, 1.8417, 1.8413], linestyle='--', color='black', zorder=PLOT_ORDER)
    plt.scatter([i for i in range(0, len(model_names))], [1.8946, 1.8139, 7.7549, 6.3596, 1.8417, 1.8413], color='red', zorder=SCATTER_ORDER)

    plt.tight_layout()
    plt.savefig("assets/mf_results/mf_results.jpg", dpi=500)

if __name__ == '__main__':
    main()