import torch
from torch.nn.utils import rnn
import matplotlib.pyplot as plt
import numpy as np


def collate_fn(batch):
    X, y, stage_lengths = zip(*batch)

    X = torch.concatenate([x.unsqueeze(0) for x in X])
    
    y = rnn.pack_sequence(y, enforce_sorted=False)
    y, y_lens = rnn.pad_packed_sequence(y, batch_first=True, padding_value=100)
    
    # sort both y and X according to y lens
    y_lens, sort_idx = y_lens.sort(descending=True)

    # sort stage_lengths according to sort_idx
    stage_lengths = [stage_lengths[i] for i in sort_idx]

    return X[sort_idx], y[sort_idx], y_lens, stage_lengths


def plot_diagnostics(model, sample, labels_dict, preds_title=None, fig_path='.'):
    X, y, y_lens, stage_lengths = sample
    model.eval()
    with torch.no_grad():
        emissions = model(X)

    id = 6

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    probs = np.exp(emissions[id].detach().numpy())
    for i in range(probs.shape[1]):
        times = np.arange(probs.shape[0])
        axs[0].plot(times, probs[:, i], label=labels_dict[i] if i != 0 else 'blank')
    #preds = [dataset.idx_to_sleep_stage[l][-1] for l in y[sum(y_lens[:id]):sum(y_lens[:id])+y_lens[id]].numpy()]
    preds = [labels_dict[l][-1] for l in y[id][:y_lens[id].item()].numpy()]

    fig.suptitle(f'True y = {preds}')
    if preds_title:
        axs[0].set_title(f'{preds_title}')
    else:
        axs[0].set_title('Predicted Probabilities')
    axs[0].set_ylabel('Probability')
    axs[0].set_xlabel('Time [windows]')
    for stage_length in np.cumsum(stage_lengths[id])[:-1]:
        axs[0].axvline(x=stage_length, color='lightgrey', linestyle='--')
        axs[1].axvline(x=stage_length, color='lightgrey', linestyle='--')
    # legend in bottom left corner
    axs[0].legend(loc='lower left')
    axs[1].plot(X[id].numpy())
    axs[1].set_title('Input signal X')
    axs[1].set_ylabel('Signals')
    axs[1].set_xlabel('Time [windows]')
    full_path = f'{fig_path}/diagnostics_{preds_title}.png'
    plt.savefig(full_path)
    plt.close()

    model.train()
    return full_path