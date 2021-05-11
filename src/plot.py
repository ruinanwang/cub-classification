import matplotlib.pyplot as plt

def plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir):
    plt.title("Loss")
    plt.plot(train_loss_list, label='Train Loss', color='green')
    plt.plot(valid_loss_list, label='Validation Loss', color='red')
    plt.legend()
    plt.savefig(save_dir+'loss.png')
    
    plt.title("Accuracy")
    plt.plot(train_acc_list, label='Train Accuracy', color='green')
    plt.plot(valid_acc_list, label='Validation Accuracy', color='red')
    plt.legend()
    plt.savefig(save_dir+'acc.png')