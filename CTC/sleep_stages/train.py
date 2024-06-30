import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn 

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

from modules import CTCNetworkCNN, CTCNetworkCNNSimple, CTCNetworkLSTM
from utils import collate_fn, plot_diagnostics


# dataset that loads data like np.load("sleep/SC4412-0.npy")
class SleepDataset(Dataset):
    def __init__(self, root_dir, transform=None, normalize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [x.split('.')[0] for x in os.listdir(root_dir) if 'labels' not in x and 'timestamps' not in x]
        
        #0 is reserved for padding... or blank?
        self.sleep_stage_to_idx = {'Sleep stage W': 7, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage R': 5, 'Sleep stage ?': 6, 'Movement time': 8}
        self.idx_to_sleep_stage = {7: 'Sleep stage W', 1: 'Sleep stage 1', 2: 'Sleep stage 2', 3: 'Sleep stage 3', 4: 'Sleep stage 4', 5: 'Sleep stage R', 6: 'Sleep stage ?', 8:'Movement time'}
        
        self.num_classes = len(self.sleep_stage_to_idx) + 1
        self.normalize = normalize
        if normalize is not None:
            self.mean = normalize[0]
            self.std = normalize[1]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        sample = os.path.join(self.root_dir, self.files[idx])
        X = np.load(sample + ".npy")
        y = np.load(sample + "-labels.npy", allow_pickle=True)
        stage_lengths = np.diff(np.load(sample + "-timestamps.npy"))
        # append the last stage length
        stage_lengths = np.append(stage_lengths, len(X) - np.sum(stage_lengths))

        if self.normalize:
            X = (X - self.mean) / self.std

        # turn y into idxs
        y = np.array([self.sleep_stage_to_idx[x] for x in y])

        # turn into torch tensors
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y)
        stage_lengths = torch.from_numpy(stage_lengths)

        return X, y, stage_lengths


def train(model, train_loader, valid_loader, epochs=10, lr = 10**(-3), sample=None, device='cpu', weight_decay=0):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                plot_diagnostics(model.to('cpu'), sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={epoch};val_loss={epoch_loss:.2f}_{wandb.run.name}', fig_path=args.fig_path)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='lstm', choices=['lstm', 'cnn', 'cnn_simple'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=10**(-3))
    parser.add_argument('--fig_path', type=str, default='') #figures/diagnostics
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--with_logging', type=bool, default=True)
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--plot_every_n_epochs', type=int, default=3)
    parser.add_argument('--initialization', type=str, default='FIR', choices=['FIR', 'FIR+He','He', 'default'])
    parser.add_argument('--tags', nargs='+')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print('Cuda is available, setting device=cuda')
            device = 'cuda'
        else:
            print('Cuda not available, setting device=cpu')
            device = 'cpu'
    else:
        device = 'cpu'

    train_set_pct = 0.8
    dataset = SleepDataset('sleep', normalize=[np.array([0, 0, 0, 0]), np.array([0.00002, 0.00001, 0.00005, 0.0000025])])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*train_set_pct), len(dataset) - int(len(dataset)*train_set_pct)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    #sample = next(iter(valid_loader))
    #with open('sample.pkl', 'wb') as f:
    #    pickle.dump(sample, f)
    #quit()

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
        config = {k:v for k, v in args.__dict__.items() if k not in ["with_logging", "tags"]}
        config['data'] = 'sleep'
           
        wandb.init(
        project="CTC", entity="metrics_logger",
        tags = args.tags,
        config=config
    )

    print(args.__dict__)
    num_features = next(iter(valid_loader))[0].shape[-1]

    if args.architecture == 'lstm':
        model = CTCNetworkLSTM(num_features=num_features, num_classes=dataset.num_classes, weight_init=args.initialization, dropout=args.dropout)
    elif args.architecture == 'cnn':
        model = CTCNetworkCNN(num_features=num_features, num_classes=dataset.num_classes, weight_init=args.initialization)
    else:
        model = CTCNetworkCNNSimple(num_features=num_features, num_classes=dataset.num_classes, kernel_size=args.kernel_size)

    print("model", model)

    losses, val_losses, model = train(model, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, sample=sample, device=device, weight_decay=args.weight_decay)

    if args.with_logging and args.fig_path:
        # save
        model.to('cpu')
        sample = next(iter(valid_loader))
        full_path = plot_diagnostics(model, sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={len(losses)};val_loss={val_losses[-1]:.2f}', fig_path=args.fig_path)
        print(full_path)
        wandb.log({"diagnostics": wandb.Image(full_path)})

        # save train.py
        wandb.save('train.py')

