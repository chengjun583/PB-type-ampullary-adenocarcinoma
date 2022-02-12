
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot
import os


def plot_scatter(Y1, Y2, Y3, Y4, Y5, Y6):
    fig,ax = plt.subplots(1,1)
    ax.set_title("SVC_linear")
    plt.ylabel('predicted probability of being CHOL')

    my_y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(my_y_ticks)

    X1 = []
    for _ in range(len(Y1)):
        X1.append(0)
    X2 = []
    for _ in range(len(Y2)):
        X2.append(1)
    X3 = []
    for _ in range(len(Y3)):
        X3.append(2)
    X4 = []
    for _ in range(len(Y4)):
        X4.append(3)
    X5 = []
    for _ in range(len(Y5)):
        X5.append(4)
    X6 = []
    for _ in range(len(Y6)):
        X6.append(5)
    s1 = plt.scatter(X1, Y1, c='steelblue', marker='o', alpha=0.8)
    s2 = plt.scatter(X2, Y2, c='mediumseagreen', marker='o', alpha=0.8)
    s3 = plt.scatter(X3, Y3, c='coral', marker='o', alpha=0.8)
    s4 = plt.scatter(X4, Y4, c='cyan', marker='o', alpha=0.8)
    s5 = plt.scatter(X5, Y5, c='red', marker='o', alpha=0.8)
    s6 = plt.scatter(X6, Y6, c='yellow', marker='o', alpha=0.8)

    index_ls = ['', 'CHOL-TCGA','CHOL-SYSUCC','PAC-TCGA','PAC-TCGA','ACC-SYSUCC','ACC-Zhejiang']
    x = range(1, 6, 1)
    plt.xticks(x, index_ls, rotation=15)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.grid(color="k", linestyle=":")

    if not os.path.exists('scatter_plot'):
        os.makedirs('scatter_plot')
    pyplot.savefig('/scatter_plot/scatter_SVC_linear.png')
    plt.show()


def main():
    # TCGA
    f1 = open("/test_results/TCGA_test_proba_SVC_linear.txt")
    line = f1.readline()
    Y1 = []
    Y3 = []
    while line:
        label = line[0:4]
        if label == 'CHOL':
            Y1.append(float(line.split('[')[1].split(' ')[0]))
        elif label == 'PAAD':
            Y3.append(float(line.split('[')[1].split(' ')[0]))
        line = f1.readline()
    f1.close()
    # SYSUCC
    f2 = open("/zs_val_results/valid_proba_SVC_linear.txt")
    line = f2.readline()
    Y2 = []
    Y4 = []
    while line:
        label = line[0:4]
        if label == 'CHOL':
            Y2.append(float(line.split('[')[1].split(' ')[0]))
        elif label == 'PAAD':
            Y4.append(float(line.split('[')[1].split(' ')[0]))
        line = f2.readline()
    f2.close()
    # SYSUCC
    f3 = open("/zs_AAC_results/AAC_proba_SVC_linear.txt")
    line = f3.readline()
    Y5 = []
    while line:
        Y5.append(float(line.split('[')[1].split(' ')[0]))
        line = f3.readline()
    f3.close()
    # Zhejiang
    f4 = open("/zj_AAC_results/AAC_proba_SVC_linear.txt")
    line = f4.readline()
    Y6 = []
    while line:
        Y6.append(float(line.split('[')[1].split(' ')[0]))
        line = f4.readline()
    f4.close()
    # Draw probability scatter diagram
    plot_scatter(Y1, Y2, Y3, Y4, Y5, Y6)


if __name__ == "__main__":
    main()
