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

from modules import CTCNetworkCNN, CTCNetworkCNNSimple, CTCNetworkLSTM
from utils import collate_fn, plot_diagnostics

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
                plot_diagnostics(model.to('cpu'), sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={epoch};val_loss={epoch_loss:.2f}', fig_path=args.fig_path)
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
    parser.add_argument('--total_samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=10**(-3))
    parser.add_argument('--fig_path', type=str, default='') #figures/diagnostics
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--with_logging', type=bool, default=False)
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
    #sample = next(iter(valid_loader))
    #with open('sample_synthetic.pkl', 'wb') as f:
    #    pickle.dump(sample, f)
    #quit()

    if args.fig_path:
        with open('sample_synthetic.pkl', 'rb') as f:
            sample = pickle.load(f)
        if not os.path.exists(args.fig_path):
            os.mkdir(args.fig_path)
    else:
        sample=None

    if args.with_logging:
        import wandb
        config = {k:v for k, v in args.__dict__.items() if k not in ["with_logging", "tags"]}
        config['data'] = 'synthetic'
           
        wandb.init(
        project="CTC", entity="metrics_logger",
        tags = args.tags,
        config=config
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
        full_path = plot_diagnostics(model, sample, dataset.idx_to_sleep_stage, preds_title=f'epoch={len(losses)};val_loss={val_losses[-1]:.2f}', fig_path=args.fig_path)
        wandb.log({"diagnostics": wandb.Image(full_path)})

        # save synthetic_train.py
        wandb.save('synthetic_train.py')

