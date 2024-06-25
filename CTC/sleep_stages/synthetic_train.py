import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn 
from torch.nn.utils import rnn
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

class SyntheticSleepDataset(Dataset):
    def __init__(self, total_samples=1000):
        #self.sleep_stage_to_idx = {'Sleep stage W': 7, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage R': 5, 'Sleep stage ?': 6, 'Movement time': 8}
        #self.idx_to_sleep_stage = {7: 'Sleep stage W', 1: 'Sleep stage 1', 2: 'Sleep stage 2', 3: 'Sleep stage 3', 4: 'Sleep stage 4', 5: 'Sleep stage R', 6: 'Sleep stage ?', 8: 'Movement time'}
        
        self.sleep_stage_to_idx = {'Class 1': 1, 'Class 2': 2, 'class 3': 3}
        self.idx_to_sleep_stage = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
    
        self.num_classes = len(self.sleep_stage_to_idx) + 1 #plus blank
        self.total_samples = total_samples
    
    def __len__(self):
        return self.total_samples
    
    @staticmethod
    def generate_signal(stage, length):
        #carrier = np.sin(np.arange(length)*stage/10)
        #carrier = (np.ones(length)*stage/10)
        frequency = stage/10 # 0.1, 0.2, or 0.3
        sr = 1 #hz
        t = np.arange(0, length, 1/sr)
        carrier = np.sin(2 * np.pi * frequency * t)
        return carrier[:, np.newaxis]
    
    def generate_sample(self, stage, length):
        carrier = self.generate_signal(stage, length)
        # return np.concatenate([carrier, carrier + 1/stage, carrier + 2/stage, carrier + 3/stage], axis=1)
        return carrier
    
    def __getitem__(self, idx):
        i = 0
        max_time = 60
        max_time = 240
        sample = np.zeros(shape=(max_time, 1))
        stages = []
        stage_lengths = []
        while i < max_time:
            stage = np.random.randint(1, self.num_classes)
            stages.append(stage)
            length = np.random.randint(5, max_time//2)
            
            if i + length > max_time:
                length = max_time - i
            stage_lengths.append(length)
            sample[i:i+length] = self.generate_sample(stage, length)
            i += length
        
        sample = torch.from_numpy(sample).float()
        stages = torch.from_numpy(np.array(stages))
        stage_lengths = torch.from_numpy(np.array(stage_lengths))
        return sample, stages, stage_lengths


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



class CTCNetworkLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_classes, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(2*num_classes, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        out = self.output(x)
        out = self.softmax(out)
        return out
    

class CTCNetworkCNN(nn.Module):
    def __init__(self, num_features, num_classes, weight_init='default'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=11, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.linear = nn.Linear(256, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

        if 'fir' in weight_init.lower():
            from scipy.signal import firwin
            numtaps = 11
            cutoff = [0.1, 0.3]
            filter_coefficients = firwin(numtaps, cutoff, pass_zero='bandpass')
            filter_coefficients = torch.tensor(filter_coefficients, dtype=torch.float32).view(1, 1, -1).repeat(32, 1, 1)

            self.conv1.weight = nn.Parameter(filter_coefficients )

        if "he" in weight_init.lower():
            nn.init.kaiming_normal_(self.linear.weight)




    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.softmax(x)

        return x
    
class CTCNetworkCNNSimple(nn.Module):
    def __init__(self, num_features, num_classes, kernel_size=11):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=kernel_size, stride=1, padding='same')
        self.linear = nn.Linear(32, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.softmax(x)

        return x


def train(model, train_loader, valid_loader, epochs=10, lr = 10**(-3), sample=None, device='cpu'):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    losses, val_losses = [], []
    num_steps = len(train_loader)*epochs
    epoch = 1
    with tqdm(range(num_steps)) as pbar:

        for step in pbar:
            if args.fig_path and (epoch % args.plot_every_n_epochs == 0 or epoch==1):
                if sample is None:
                    sample = X, y, y_lens, stage_lengths
                epoch_loss = val_losses[-1] if val_losses else np.nan
                plot_diagnostics(model.to('cpu'), sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={epoch};val_loss={epoch_loss:.2f}')
                model.to(device)
        
            X, y, y_lens, stage_lengths = next(iter(train_loader))
            X, y, y_lens, = X.to(device), y.to(device), y_lens.to(device)

            optimizer.zero_grad()
            emissions = model(X)
            input_lengths = torch.full(size=(X.shape[0],), fill_value=X.shape[1], dtype=torch.long, device=device)
            loss = ctc_loss(emissions.permute(1, 0, 2), y, input_lengths, y_lens, )
            loss.backward()
            optimizer.step()

            # Report
            if step % 20 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step :5}, loss={loss:.2f}")

            if (step+1) % len(train_loader) == 0:
                epoch += 1
                losses.append(loss.item())
                # record loss with wandb
                if args.with_logging:
                    wandb.log({'train_loss': loss.item()})
                model.eval()
                with torch.no_grad():
                    X, y, y_lens, stage_lengths = next(iter(valid_loader))
                    X, y, y_lens, = X.to(device), y.to(device), y_lens.to(device)
                    val_loss = ctc_loss(model(X).permute(1, 0, 2), y, input_lengths, y_lens)
                    if args.with_logging:
                        wandb.log({'val_loss': val_loss.item()})
                    val_losses.append(val_loss.item())
                model.train()

        return losses, val_losses, model



def plot_diagnostics(model, sample, labels_dict, preds_title=None):
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
    full_path = f'{args.fig_path}/diagnostics_{preds_title}.png'
    plt.savefig(full_path)
    plt.close()

    model.train()
    return full_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='lstm', choices=['lstm', 'cnn', 'cnn_simple'])
    parser.add_argument('--total_samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=10**(-3))
    parser.add_argument('--fig_path', type=str, default='') #figures/diagnostics
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--with_logging', type=bool, default=True)
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--plot_every_n_epochs', type=int, default=3)
    parser.add_argument('--initialization', type=str, default='FIR', choices=['FIR', 'FIR+He','He', 'default'])
    parser.add_argument('--tags', nargs='+')

    args = parser.parse_args()
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print('CUDA')
            device = 'cuda'
        else:
            print('Cuda not available, setting device=cpu')
            device = 'cpu'
    else:
        device = 'cpu'

    train_set_pct = 0.8
    dataset = SyntheticSleepDataset(total_samples=args.total_samples)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*train_set_pct), len(dataset) - int(len(dataset)*train_set_pct)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    # load sample.pkl from pickle
    if args.fig_path:
        with open('sample.pkl', 'rb') as f:
            sample = pickle.load(f)
        if not os.path.exists(args.fig_path):
            os.mkdir(args.fig_path)
    else:
        sample=None

    if args.with_logging:
        import wandb
           
        wandb.init(
        project="CTC", entity="metrics_logger",
        tags = args.tags,
        config={k:v for k, v in args.__dict__.items() if k not in ["with_logging", "tags"]}
    )

    print(args.__dict__)

    if args.architecture == 'lstm':
        model = CTCNetworkLSTM(num_features=1, num_classes=dataset.num_classes)
    elif args.architecture == 'cnn':
        model = CTCNetworkCNN(num_features=1, num_classes=dataset.num_classes, weight_init=args.initialization)
    else:
        model = CTCNetworkCNNSimple(num_features=1, num_classes=dataset.num_classes, kernel_size=args.kernel_size)

    print("model", model)

    losses, val_losses, model = train(model, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, sample=sample, device=device)

    if args.with_logging and args.fig_path:
        # save
        model.to('cpu')
        sample = next(iter(valid_loader))
        full_path = plot_diagnostics(model, sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={len(losses)};val_loss={val_losses[-1]:.2f}')
        wandb.log({"diagnostics": wandb.Image(full_path)})

        # save synthetic_train.py
        wandb.save('synthetic_train.py')

