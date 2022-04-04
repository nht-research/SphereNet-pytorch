import argparse
from spherenet import OmniMNIST, OmniFashionMNIST
from spherenet import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms


class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        self.conv1 = SphereConv2D(1, 32, stride=1)
        self.pool1 = SphereMaxPool2D(stride=2)
        self.conv2 = SphereConv2D(32, 64, stride=1)
        self.pool2 = SphereMaxPool2D(stride=2)

        self.fc = nn.Linear(14400, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 14400)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64 * 13 * 13, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 13 * 13)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return x


class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)  # flatten, [B, C, H, W) -> (B, C*H*W)
        x = self.fc(x)
        return x


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if data.dim() == 3:
                data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='MNIST',
                        help='dataset for training, options={"FashionMNIST", "MNIST"}')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer, options={"adam, sgd"}')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before saving model weights')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    np.random.seed(args.seed)
    if args.data == 'FashionMNIST':
        test_dataset = OmniFashionMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, img_std=255, train=False,
                                        fix_aug=True)
    elif args.data == 'MNIST':
        test_dataset = OmniMNIST(fov=120, flip=True, h_rotate=True, v_rotate=True, train=False, fix_aug=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    load_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    perspective_test_dataset = datasets.MNIST('datas/MNIST', train=False, download=True, transform=load_transform)
    perspective_test_loader = torch.utils.data.DataLoader(perspective_test_dataset, batch_size=args.test_batch_size,
                                                          shuffle=False, **kwargs)


    perspective_model = NetMNIST()
    perspective_model_state_dict = torch.load(
        '/home/iago/workspace/SphereNet-pytorch/datas/models/model_perspective.pkl')
    perspective_model.load_state_dict(perspective_model_state_dict)
    perspective_model = perspective_model.to(device).eval()

    sphere_model = SphereNet()
    sphere_state_dict = torch.load('/home/iago/workspace/SphereNet-pytorch/datas/models/sphere_model.pkl')
    sphere_model.load_state_dict(sphere_state_dict)
    sphere_model = sphere_model.to(device).eval()

    model = Net()
    model_state_dict = torch.load('/home/iago/workspace/SphereNet-pytorch/datas/models/model.pkl')
    model.load_state_dict(model_state_dict)
    model = model.to(device).eval()

    # SphereCNN
    print('{} Sphere CNN {}'.format('=' * 10, '=' * 10))
    test(args, sphere_model, device, test_loader)

    # Conventional CNN
    print('{} Conventional CNN {}'.format('=' * 10, '=' * 10))
    test(args, model, device, test_loader)

    # Perspective CNN
    print('{} Perspective CNN {}'.format('=' * 10, '=' * 10))
    test(args, perspective_model, device, perspective_test_loader)


if __name__ == '__main__':
    main()
