from matplotlib import pyplot as plt


def plot_model_loss(history_dict):

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss", color='red')
    plt.plot(epochs, val_loss_values, "b", label="Validation loss", color='blue')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_model_accuracy(history_dict):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc", color='red')
    plt.plot(epochs, val_acc, "b", label="Validation acc", color='blue')
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()