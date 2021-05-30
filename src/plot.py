import matplotlib.pyplot as plt

def plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir):
    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    
    ax1.set_title("Loss")
    ax1.plot(train_loss_list, label='Train Loss', color='green')
    ax1.plot(valid_loss_list, label='Validation Loss', color='red')
    ax1.legend()
    fig1.savefig(save_dir+'loss.png')
    
    ax2.set_title("Accuracy")
    ax2.plot(train_acc_list, label='Train Accuracy', color='green')
    ax2.plot(valid_acc_list, label='Validation Accuracy', color='red')
    ax2.legend()
    fig2.savefig(save_dir+'acc.png')
    
# def plot_loss(train, valid, save_dir):
#     plt.title("Loss")
#     plt.plot(train, label='Train Loss', color='green')
#     plt.plot(valid, label='Validation Loss', color='red')
#     plt.legend()
#     plt.savefig(save_dir+'loss.png')
    
# def plot_acc(train, valid, save_dir):
#     plt.title("Loss")
#     plt.plot(train, label='Train Accuracy', color='green')
#     plt.plot(valid, label='Validation Accuracy', color='red')
#     plt.legend()
#     plt.savefig(save_dir+'acc.png')