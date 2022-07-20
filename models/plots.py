import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_model_loss(history_dict, name):

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "ro", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("LOSS - " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('plots/experiments/loss_' + name + '.png')
    plt.clf()
    # plt.show()


def plot_model_accuracy(history_dict, name):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, acc, "ro", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("ACCURACY - " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('plots/experiments/accuracy_' + name + '.png')
    plt.clf()
    # plt.show()