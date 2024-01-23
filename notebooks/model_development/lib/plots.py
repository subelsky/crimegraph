import matplotlib.pyplot as plt

def plot_loss_and_metrics(history, name):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title(f'{name} Loss')
    plt.xlim(0, len(history['loss'])-1)

    plt.subplot(2, 1, 2)
    plt.plot(history['hit_rate'], label='Hit Rate Accuracy')
    plt.plot(history['val_hit_rate'], label='Validation Hit Rate Accuracy')
    plt.title('Model Hit Rate')
    plt.ylabel('Hit Rate')
    plt.xlabel('Epoch')
    plt.title(f'{name} Hit Rate')
    plt.xlim(0, len(history['hit_rate'])-1)
    plt.legend()

    plt.tight_layout()
    plt.show()