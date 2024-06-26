import torch.nn as nn 
import torch

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