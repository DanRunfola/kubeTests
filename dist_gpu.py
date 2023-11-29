import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import socket

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def acquire_lock(lock_file, timeout=300):
    start_time = time.time()
    while True:
        try:
            # Try to create a lock file
            with os.fdopen(os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY), 'w') as f:
                f.write(f"Lock acquired by {os.environ['HOSTNAME']}\n")
            break
        except FileExistsError:
            # If lock file exists, check the timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout while waiting for file lock: {lock_file}")
            time.sleep(1)  # Wait for a while before retrying

def release_lock(lock_file):
    try:
        os.remove(lock_file)
    except FileNotFoundError:
        pass  # Ignore if the file was already removed

def setup_distributed(claims_dir, rank, world_size):
    lock_file = os.path.join(claims_dir, 'dist_lock')
    claims_file = os.path.join(claims_dir, 'claims.json')

    acquire_lock(lock_file)

    try:
        if os.path.exists(claims_file):
            with open(claims_file, 'r') as f:
                claims = json.load(f)
        else:
            claims = {'master_addr': None, 'ranks': {}}

        if claims['master_addr'] is None:
            # Current pod becomes the master
            claims['master_addr'] = socket.gethostbyname(socket.gethostname())

        if os.environ['HOSTNAME'] not in claims['ranks']:
            claims['ranks'][os.environ['HOSTNAME']] = rank

        with open(claims_file, 'w') as f:
            json.dump(claims, f)
    finally:
        release_lock(lock_file)

    master_addr = claims['master_addr']
    rank = claims['ranks'][os.environ['HOSTNAME']]

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main():
    claims_dir = '/kube/home/.claims/'
    os.makedirs(claims_dir, exist_ok=True)

    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    setup_distributed(claims_dir, rank, world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='/kube/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Train the model
    for epoch in range(1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {loss.item()}')

    cleanup()

if __name__ == '__main__':
    main()
