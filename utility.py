import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
import sys
import os
import pickle

from tensorflow.keras.models import load_model, clone_model

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'lime', 'violet']
# linestyles = []


def parseArg():
    parser = argparse.ArgumentParser(description='Plot results. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for FedMD.'
                        )

    conf_file = os.path.abspath("conf/EMNIST_balance_conf.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


def plot_history(model):
    
    """
    input : model is trained keras model.
    """
    
    fig, axes = plt.subplots(2,1, figsize = (12, 6), sharex = True)
    axes[0].plot(model.history.history["loss"], "b.-", label = "Training Loss")
    axes[0].plot(model.history.history["val_loss"], "k^-", label = "Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(model.history.history["acc"], "b.-", label = "Training Acc")
    axes[1].plot(model.history.history["val_acc"], "k^-", label = "Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.subplots_adjust(hspace=0)
    plt.show()


def show_performance(model, Xtrain, ytrain, Xtest, ytest):
    y_pred = None
    print("CNN+fC Training Accuracy :")
    y_pred = model.predict(Xtrain, verbose = 0).argmax(axis = 1)
    print((y_pred == ytrain).mean())
    print("CNN+fc Test Accuracy :")
    y_pred = model.predict(Xtest, verbose = 0).argmax(axis = 1)
    print((y_pred == ytest).mean())
    print("Confusion_matrix : ")
    print(confusion_matrix(y_true = ytest, y_pred = y_pred))
    
    del y_pred


if __name__ == "__main__":
    conf_file = parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())

        N_parties = conf_dict["N_parties"]
        model_saved_dir = conf_dict["model_saved_dir"]
        result_save_dir = "./FEMNIST_balanced_MNIST/"
        comp_result_save_dir = "./FEMNIST_balanced_Fashion_MNIST/"

    with open(os.path.join(comp_result_save_dir,'1207/col_performance_MNIST_30_50.pkl'), 'rb') as f_baseline:
        baseline_MNIST = pickle.load(f_baseline, encoding='bytes')
        print(baseline_MNIST)
        f_baseline.close()

    with open(os.path.join(comp_result_save_dir,'1207/col_performance_Fashion_MNIST_30_50.pkl'), 'rb') as f2:
        baseline_Fashion_MNIST = pickle.load(f2, encoding='bytes')
        print(baseline_Fashion_MNIST)
        f2.close()
    print(baseline_Fashion_MNIST)

    with open(os.path.join(comp_result_save_dir,'1207/col_performance_gan_without_MNIST_30_50.pkl'), 'rb') as f:
        gan = pickle.load(f, encoding='bytes')
        print(gan)
        f.close()

    with open(os.path.join(comp_result_save_dir,'1207/col_performance_gan_Fashion_MNIST_30_50.pkl'), 'rb') as f1:
        gan_Fashion_MNIST = pickle.load(f1, encoding='bytes')
        print(gan_Fashion_MNIST)
        f1.close()

    with open(os.path.join(comp_result_save_dir, '1207/col_performance_gan_MNIST_30_50.pkl'), 'rb') as f3:
        gan_MNIST = pickle.load(f3, encoding='bytes')
        print(gan_MNIST)
        f3.close()

    # for i in range(len(data)):
    #     print("model ", i)
    #     print(data[i])
    #     if i == 0 or 4 or 5 or 9:
    # plt.title('Test Accuracy vs. Epochs (on EMNIST)')
    # plt.plot(range(1,len(data[0][:80])+1), data[0][:80], color='g', linestyle='-', marker='o', label='MNIST, model '+str(0))
    # plt.plot(range(1, 15), data_comp[0][:14][:80], color='g', linestyle='--', marker='<', label='Fashion_MNIST, model ' + str(0))

    plt.plot(range(1, 32), baseline_MNIST[1][:80], color='m', linestyle='-', marker='>', linewidth=1.0, ms=5, label='MNIST, model ' + str(1))
    plt.plot(range(1, 32), baseline_Fashion_MNIST[1][:80], color='g', linestyle='--', marker='s', linewidth=1.0, ms=5, label='Fashion_MNIST, model ' + str(1))
    plt.plot(range(1, 32), gan[1][:80], color='b', linestyle='-', marker='x', linewidth=1.0, ms=6, label='GAN, model ' + str(1))
    plt.plot(range(1, 32), gan_Fashion_MNIST[1][:80], color='r', linestyle='--', marker='o', linewidth=1.0, ms=5, label='Fashion_MNIST+GAN, model ' + str(1))
    # plt.plot(range(1, 15), data_comp1126[2][:80], color='y', linestyle='-.', marker='o', linewidth=1.0, ms=5, label='Fashion_MNIST+GAN, model ' + str(2))
    # plt.plot(range(1, 15), data_comp1126[3][:80], color='c', linestyle='-.', marker='o', linewidth=1.0, ms=5, label='Fashion_MNIST+GAN, model ' + str(3))

    plt.plot(range(1, 32), gan_MNIST[1][:80], color='black', linestyle='-.', marker='p', linewidth=1.0, ms=5, label='MNIST+GAN, model ' + str(1))

    # plt.plot(range(1,len(data_comp[2][:80]) + 1), data_comp[2][:80], color='c', label='model' + str(2))
    # plt.plot(range(1, len(data_comp[2][:80]) + 1), data_comp[2][:80], color='c', label='model' + str(0))
    # plt.plot(range(1,len(data[3]) + 1), data[3], color='y', label='model' + str(3))
    # plt.plot(range(1, len(data[4]) + 1), data[4], color='y', label='model' + str(4))
    # plt.plot(range(1,len(data[5])+1), data[5], color='r', linestyle='-', marker='o', label='MNIST, model ' + str(5))
    # plt.plot(range(1, 15), data_comp[5][:14][:80], color='r', linestyle='--', marker='<', label='Fashion_MNIST, model ' + str(5))
    # plt.plot(range(1, len(data[6]) + 1), data[6], color='y', label='model' + str(6))
    # plt.plot(range(1, len(data[7]) + 1), data[7], color='m', linestyle='-', marker='x', linewidth=1.0, ms=6, label='GAN, model' + str(7))
    # plt.plot(range(1, len(data_comp[7]) + 1), data_comp[7], color='m', linestyle='--', marker='s', linewidth=1.0, ms=6, label='Fashion_MNIST, model' + str(7))

    # plt.plot(range(1, len(data[8][:80]) + 1), data[8][:80], color='g', linestyle='-', marker='x', linewidth=1.0, ms=6, label='GAN, model ' + str(8))
    # plt.plot(range(1, 15), data_comp[8][:80], color='g', linestyle='--', marker='s', linewidth=1.0, ms=5, label='Fashion_MNIST, model ' + str(8))
    # plt.plot(range(1,len(data[9][:80])+1), data[9][:80], color='r', linestyle='-', marker='x', linewidth=1.0, ms=6, label='GAN, model '+str(9))
    # plt.plot(range(1, 15), data_comp[9][:80], color='r', linestyle='--', marker='s', linewidth=1.0, ms=5, label='Fashion_MNIST, model '+str(9))
    # plt.plot(range(1, len(data_comp[9][:80]) + 1), data_comp[9][:80], color='g', label='Fashion_MNIST, model' + str(9))
    # plt.axhline(y=upper_bound[9]['val_acc'][-1], color='m', linestyle='-')

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()

    plt.subplots_adjust(hspace=0)
    plt.savefig(result_save_dir+"acc-gan_1228.png")
    plt.show()

