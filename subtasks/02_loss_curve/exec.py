import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.figure(figsize=(8, 4))

    mf_loss = np.load(f"saved/loss/train_loss_mf.book.npy")
    epochs = np.array([i + 1 for i in range(0, len(mf_loss))])
    plt.plot(epochs, mf_loss, color='blue', label="MF")

    mf_bias_loss = np.load(f"saved/loss/train_loss_mf_bias.book.npy")
    epochs = np.array([i + 1 for i in range(0, len(mf_bias_loss))])
    plt.plot(epochs, mf_bias_loss, color='red', label="MF+bias")

    ncf_loss = np.load(f"saved/loss/train_loss_ncf.book.npy")
    epochs = np.array([i + 1 for i in range(0, len(ncf_loss))])
    plt.plot(epochs, ncf_loss, color='blue', label="Neural CF")

    ncf_bias_loss = np.load(f"saved/loss/train_loss_ncf_bias.book.npy")
    epochs = np.array([i + 1 for i in range(0, len(ncf_bias_loss))])
    plt.plot(epochs, ncf_bias_loss, color='red', label="Neural CF+bias")

    plt.title(f"Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")

    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"assets/loss_curve/loss_curve.jpg", dpi=500)

if __name__ == '__main__':
    main()