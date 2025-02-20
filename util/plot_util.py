import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

def plot_loss(lightning_log_version, batch_per_epoch, loss_types=['train_loss', 'val_loss', 'test_loss'], show=True, save_png_path=None):
    event_acc = EventAccumulator(lightning_log_version)
    event_acc.Reload()

    loss_vals = {} # {'loss_name':[step1_loss, step2_loss, ... ]}
    steps = {} # {'loss_name':[step1, step2, ... ]}

    for lt in loss_types:
        try:
            loss_events = event_acc.Scalars(lt)
            loss_vals[lt] = [e.value for e in loss_events]
            steps[lt] = [e.step for e in loss_events]
        except:
            print(f"warning: {lt} is empty")
            pass

    
    for loss_type, step in steps.items():
        loss_val = loss_vals[loss_type]

        epochs = []
        epoch_losses = []

        curr_epoch = 1
        curr_loss_sum = 0
        curr_step_sum = 0
        for s, l in zip(step, loss_val):
            s += 1
            curr_loss_sum += l
            curr_step_sum += 1

            if batch_per_epoch == 1 or (s // batch_per_epoch == curr_epoch):
                avg_loss = curr_loss_sum / curr_step_sum
                true_epoch = s / batch_per_epoch
                epochs.append(true_epoch)
                epoch_losses.append(avg_loss)

                curr_epoch += 1
                curr_loss_sum = 0
                curr_step_sum = 0
        if curr_step_sum > 0:
            avg_loss = curr_loss_sum / curr_step_sum
            true_epoch = step[-1] / batch_per_epoch
            epochs.append(true_epoch)
            epoch_losses.append(avg_loss)

        plt.plot(epochs, epoch_losses, label=loss_type)

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss')
    plt.legend()

    if save_png_path:
        plt.savefig(save_png_path)

    if show:
        plt.show()


def plot_loss_last_version(lightning_log='lightning_logs', batch_per_epoch=1, loss_types=['train_loss', 'val_loss', 'test_loss'], show=True, save_png_path=None):
    data = glob.glob(os.path.join(lightning_log, 'version_*'))
    target_version = sorted(data, key=lambda x: int(x.split("_")[-1]))[-1]
    print(target_version)
    plot_loss(target_version, batch_per_epoch, loss_types, show, save_png_path)


if __name__ == '__main__':
    plot_loss_last_version(
        lightning_log='/home/lutao/workspace/d2learning/lec4_softmax_regression/lightning_logs',
        batch_per_epoch=235)