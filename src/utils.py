import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
import argparse
import random as rm
import math
import sys
import time
import numpy as np

from src.resnet import *
from src.vgg import *


def parse_arguments():

    # Parse the imput parameters
    parser = argparse.ArgumentParser(description='CIFAR testing')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='SSA epsilon')
    parser.add_argument('--restart_epsilon', type=float, default=-1,
                        help='Restart epsilon')
    parser.add_argument('--accepted_bound', type=float, default=0.6,
                        help='Bound accepted ratio')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature')
    parser.add_argument('--cooling_factor', type=float, default=0.97,
                        help='Cooling factor alpha')
    parser.add_argument('--numworkers', type=int, default=1,
                        help='Numworkers for the loading class (default: 1)')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='Select the GPU on the cluster')
    parser.add_argument('--net', dest="net", action="store", default="vgg16",
                        help='Network architecture to be used')
    parser.add_argument('--train_mode', dest="train_mode", action="store", default="sgd",
                        help='Train mode')
    parser.add_argument('--dataset', dest="dataset", action="store", default="cifar10",
                        help='Dataset on which to train the network')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--random-seed', dest='random_seed', action='store_true', default=False)
    parser.add_argument('--outfolder', type=str, help='Out folder for results')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use')
    parser.add_argument('--lr_set', type=str, help='LR set to use in SGD-SA')
    parser.add_argument('--dataum', dest='dataum', action='store_true', default=False)
    parser.add_argument('--cluster', action='store_true', default=False)

    return parser.parse_args()


def set_device(gpu_number):
    torch.cuda.set_device(gpu_number)


def set_seed(seed, random=False):

    rm.seed(seed)

    if random:
        for i in range(1000):
            seed = rm.randint(-10e12, 10e12)

    rm.seed(seed)

    # Reproducible runs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    return seed


def choose_optimizer(model, args):

    opt = args.optimizer

    if opt == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)
    elif opt == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters())
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif opt == 'adamax':
        optimizer = optim.Adamax(model.parameters())
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())

    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(train_loader, model, optimizer, criterion):
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        del inputs, labels, outputs


def load_dataset(dataset_name="cifar10", minibatch=512, num_workers=2, dataum=False, drop_last=False):
    if dataset_name == "cifar10":
        if dataum:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train_set = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True,
                                                 download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=minibatch,
                                                   shuffle=True, num_workers=num_workers, drop_last=drop_last)
        test_set = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False,
                                                download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        return train_loader, test_loader
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))])
        trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100',
                                                 train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        testset = torchvision.datasets.CIFAR100(root='../../data/cifar100',
                                                train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return trainloader, testloader
    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='../../data/mnist', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch,
                                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
        testset = torchvision.datasets.MNIST(root='../../data/mnist', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch,
                                                 shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return trainloader, testloader
    elif dataset_name == "fashion-mnist":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.FashionMNIST(root='../../data/fashion-mnist',
                                                          train=True, download=True, transform=transform)

        test_dataset = torchvision.datasets.FashionMNIST(root='../../data/fashion-mnist',
                                                         train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=minibatch,
                                                   shuffle=True, num_workers=num_workers, drop_last=drop_last)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=minibatch,
                                                  shuffle=False, num_workers=num_workers, drop_last=drop_last)
        return train_loader, test_loader


def load_net(net="alexnet", dataset_name="cifar10"):

    num_classes = 10
    if dataset_name == "cifar10" or dataset_name == "mnist" or dataset_name == "fashion-mnist":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "imagenet":
        num_classes = 1000

    if net == "resnet18":
        return ResNet18()
    elif net == "resnet34":
        return ResNet34()
    elif net == "resnet50":
        return ResNet50()
    elif net == "resnet101":
        return ResNet101()
    elif net == "vgg11":
        return VGG("VGG11", num_classes)
    elif net == "vgg16":
        return VGG("VGG16", num_classes)
    elif net == "vgg19":
        return VGG("VGG19", num_classes)


def test(test_loader, model):

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in test_loader:
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, labels.cuda()).item())
        del images, labels, outputs

    return test_loss / total, correct / total


def test_train(train_loader, model):

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    for data in train_loader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / total, correct / total


def test_minibatch(images, labels, model):

    model.eval()

    correct = 0
    total = labels.size(0)
    test_loss = 0

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
    test_loss += float(F.cross_entropy(outputs, labels).item())

    model.train()

    return test_loss / total, correct / total


def test_train_sample(train_loader, model, n_minibatches=10):

    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    length = 0

    train_iter = iter(train_loader)
    indexes = [0]

    while len(indexes) - 1 != n_minibatches:
        num = rm.randint(0, len(train_iter) - 2)  # Skip last minibatch
        if num not in indexes:
            indexes.append(num)

    indexes = sorted(indexes)

    for i in range(1, n_minibatches + 1):
        for j in range(indexes[i] - indexes[i - 1] - 1):
            train_iter.next()

        images, labels = train_iter.next()
        length += len(images)

        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels.cuda()).sum().item()
        test_loss += float(F.cross_entropy(outputs, Variable(labels.cuda())).item())
        del images, labels, outputs

    return test_loss / total, correct / total


def save_model(model, epoch, path):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)


def train_sgd(train_loader, test_loader, model, args):
    print('Training with SGD')

    val_loss, val_acc, train_loss, train_acc, times = [['' for i in range(args.epochs)] for j in range(5)]
    start = time.clock()

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, args)
    if args.train_mode == 'scheduled_sgd':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.1)

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)

        if args.train_mode == 'scheduled_sgd':
            scheduler.step()

        train(train_loader, model, optimizer, criterion)
        val_loss[epoch], val_acc[epoch] = test(test_loader, model)
        print("Validation loss: ", val_loss[epoch])
        print("Validation accuracy: ", val_acc[epoch])

        train_loss[epoch], train_acc[epoch] = test_train_sample(train_loader, model)
        print("Training loss: ", train_loss[epoch])
        print("Training accuracy:", train_acc[epoch])

        times[epoch] = time.clock() - start

        # Flush the output on svrnvidia
        sys.stdout.flush()
        sys.stderr.flush()

    np.savez(f'results_{args.train_mode}__seed={args.seed}_random-seed={args.random_seed}_'
             f'data-aug={args.dataum}_{args.optimizer}_{args.dataset}_{args.net}'
             f'_{args.batch_size}_{args.epochs}_lr={args.lr}_eps-restart={args.restart_epsilon}',
             times=times,
             validation_accuracy=val_acc,
             validation_loss=val_loss,
             train_accuracy=train_acc,
             train_loss=train_loss,
             epochs=np.array([args.epochs]),
             lr=np.array([args.lr]),
             momentum=np.array([args.momentum]),
             epsilon=args.restart_epsilon)

    del model


def stochastic_simulated_annealing(train_loader, model, epsilon, T):
    model.train()

    not_accepted, accepted = 0, 0
    probabilities = []

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        initial_loss = test_minibatch(inputs, labels, model)[0]

        # List used to keep the move to get back to the initial point
        inverse = []

        # First move
        for param in model.parameters():
            # Replicate the tensor
            tensor_size = param.data.size()
            move = torch.zeros(tensor_size)
            # Send it to the GPU
            move = move.cuda()
            # Generate move
            move = move.normal_(std=epsilon)
            # Step back is saved
            inverse.append(move.mul(-1))
            # Move the parameters
            param.data.add_(move)
        # Evaluate the loss
        first_loss = test_minibatch(inputs, labels, model)[0]

        # Second move
        for k, param in enumerate(model.parameters()):
            param.data.add_(inverse[k].mul(2))
            inverse[k].mul_(-1)
        second_loss = test_minibatch(inputs, labels, model)[0]

        # Get back if the first move is better
        if first_loss < second_loss:
            for k, param in enumerate(model.parameters()):
                param.data.add_(inverse[k].mul(2))
                inverse[k].mul_(-1)
            new_loss = first_loss
        else:
            new_loss = second_loss

        if new_loss > initial_loss:
            probabilities.append(math.exp(- (new_loss - initial_loss) / T))

        # Reject worse solution according to the metropolis formula
        if new_loss > initial_loss and math.exp(- (new_loss - initial_loss) / T) < rm.random():
            not_accepted += 1
            for k, param in enumerate(model.parameters()):
                param.data.add_(inverse[k])
            new_loss = initial_loss
        elif new_loss > initial_loss:
            accepted += 1

        del move, inverse, inputs, labels

    return not_accepted, accepted, probabilities


def train_ssa(train_loader, test_loader, model, args):
    val_loss, val_acc, train_loss, train_acc, times = [['' for i in range(args.epochs)] for j in range(5)]
    probabilities, na, ac, drop_eps, bm = {}, [], [], {}, []
    start = time.clock()

    epsilon = args.epsilon

    temperature = args.temperature
    cooling_factor = args.cooling_factor

    print("Bound:", args.accepted_bound)
    for epoch in range(args.epochs):
        print("Epoch: ", epoch, ", epsilon: ", epsilon, ", temperature:", temperature)

        not_accepted, accepted, probs = stochastic_simulated_annealing(train_loader, model, epsilon, temperature)

        probabilities[epoch] = probs
        na.append(not_accepted)
        ac.append(accepted)
        bm.append((len(train_loader)-(accepted+not_accepted)) / len(train_loader))
        times[epoch] = time.clock() - start

        # Training information
        print(f"Better moves: {len(train_loader)-(accepted+not_accepted)}/{len(train_loader)},"
              f" Accepted: {accepted}, Not accepted: {not_accepted}")

        # Evaluation on training set
        train_loss[epoch], train_acc[epoch] = test_train_sample(train_loader, model)
        print("Training loss: ", train_loss[epoch])
        print("Training accuracy:", train_acc[epoch])

        # Evaluation on validation set
        val_loss[epoch], val_acc[epoch] = test(test_loader, model)
        print("Validation loss: ", val_loss[epoch])
        print("Validation accuracy: ", val_acc[epoch])

        temperature *= cooling_factor
        temperature = max(1e-10, temperature)
        accepted_ratio = 1 - (not_accepted + accepted) / len(train_loader)

        print(f"Accepted ratio:{accepted_ratio}")
        if accepted_ratio < args.accepted_bound:
            drop_eps[epoch] = True
            epsilon /= 10

        # Flush the output on svrnvidia
        sys.stdout.flush()
        sys.stderr.flush()

    np.savez(f'results_{args.train_mode}_data-aug={args.dataum}_opt={args.optimizer}_{args.dataset}_{args.net}'
             f'_batch-size={args.batch_size}_epochs={args.epochs}_bound={args.accepted_bound}_eps={args.epsilon}',
             times=times,
             validation_accuracy=val_acc,
             validation_loss=val_loss,
             train_accuracy=train_acc,
             train_loss=train_loss,
             epochs=np.array([args.epochs]),
             lr=np.array([args.lr]),
             momentum=np.array([args.momentum]),
             probabilities=probabilities,
             na=na,
             ac=ac,
             better_moves=bm,
             drop_eps=drop_eps,
             epsilon=args.epsilon)


def SGD_SA(train_loader, model, optimizer, criterion, args, T):
    not_accepted, accepted = 0, 0
    probabilities = []

    for i, data in enumerate(train_loader, 0):

        if args.lr_set == "set1":
            lr = rm.choice([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                            0.1, 0.09, 0.08, 0.07, 0.06, 0.05])
        elif args.lr_set == "set2":
            lr = rm.choice([0.5, 0.4, 0.3, 0.2,
                            0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
        elif args.lr_set == "set3":
            values = np.arange(0.1, 1, 0.1)
            lr = rm.choice(values)
        elif args.lr_set == "set4":
            values = np.arange(0.01, 0.1, 0.01)
            lr = rm.choice(values)
        elif args.lr_set == "set5":
            values = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1)])
            lr = rm.choice(values)

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        initial_loss = test_minibatch(inputs, labels, model)[0]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        for name, param in model.named_parameters():
            param.data.sub_(param.grad.data.mul(lr))

        new_loss = test_minibatch(inputs, labels, model)[0]

        if new_loss > initial_loss:
            probabilities.append(math.exp(- (new_loss - initial_loss) / T))

        if new_loss > initial_loss and math.exp(- (new_loss - initial_loss) / T) < rm.random():
            not_accepted += 1
            for k, f in enumerate(model.parameters(), 0):
                f.data.add_(f.grad.data.mul(lr))
        elif new_loss > initial_loss:
            accepted += 1

        del inputs, labels, outputs

    return not_accepted, accepted, probabilities


def SGD_SA_acc(train_loader, model, optimizer, criterion, args, T):
    not_accepted, accepted = 0, 0
    probabilities = []

    for i, data in enumerate(train_loader, 0):

        if args.lr_set == "set1":
            lr = rm.choice([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                            0.1, 0.09, 0.08, 0.07, 0.06, 0.05])
        elif args.lr_set == "set2":
            lr = rm.choice([0.5, 0.4, 0.3, 0.2,
                            0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
        elif args.lr_set == "set3":
            values = np.arange(0.1, 1, 0.1)
            lr = rm.choice(values)
        elif args.lr_set == "set4":
            values = np.arange(0.01, 0.1, 0.01)
            lr = rm.choice(values)
        elif args.lr_set == "set5":
            values = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1)])
            lr = rm.choice(values)

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        initial_accuracy = test_minibatch(inputs, labels, model)[1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        for name, param in model.named_parameters():
            param.data.sub_(param.grad.data.mul(lr))

        new_accuracy = test_minibatch(inputs, labels, model)[1]

        if new_accuracy < initial_accuracy:
            probabilities.append(math.exp((new_accuracy - initial_accuracy) / T))

        if new_accuracy < initial_accuracy and math.exp((new_accuracy - initial_accuracy) / T) < rm.random():
            not_accepted += 1
            for k, f in enumerate(model.parameters(), 0):
                f.data.add_(f.grad.data.mul(lr))
        elif new_accuracy < initial_accuracy:
            accepted += 1

        del inputs, labels, outputs

    return not_accepted, accepted, probabilities


def train_SGD_SA(train_loader, test_loader, model, args):
    val_loss, val_acc, train_loss, train_acc, times = [['' for i in range(args.epochs)] for j in range(5)]
    probabilities, na, ac, drop_eps, bm = {}, [], [], {}, []
    start = time.clock()

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, args)

    temperature = args.temperature
    cooling_factor = args.cooling_factor

    for epoch in range(args.epochs):
        print("Epoch: ", epoch, ", temperature:", temperature)

        if args.train_mode == "sgd_sa":
            not_accepted, accepted, probs = SGD_SA(train_loader, model, optimizer, criterion, args, temperature)
        elif args.train_mode == 'sgd_sa_acc':
            not_accepted, accepted, probs = SGD_SA_acc(train_loader, model, optimizer, criterion, args, temperature)

        probabilities[epoch] = probs
        na.append(not_accepted)
        ac.append(accepted)

        # Training information
        print(f"Better moves: {len(train_loader)-(accepted+not_accepted)}/{len(train_loader)},"
              f" Accepted: {accepted}, Not accepted: {not_accepted}")

        # Evaluation on the training set
        train_loss[epoch], train_acc[epoch] = test_train_sample(train_loader, model)
        print("Training loss: ", train_loss[epoch])
        print("Training accuracy:", train_acc[epoch])

        # Evaluation on the validation set
        val_loss[epoch], val_acc[epoch] = test(test_loader, model)
        print("Validation loss: ", val_loss[epoch])
        print("Validation accuracy: ", val_acc[epoch])

        temperature *= cooling_factor
        temperature = max(1e-14, temperature)

        times[epoch] = time.clock() - start

    np.savez(f'results_{args.train_mode}_seed={args.seed}_random-seed={args.random_seed}_'
             f'set={args.lr_set}_data-aug={args.dataum}_opt={args.optimizer}_{args.dataset}_'
             f'{args.net}_{args.batch_size}_epochs={args.epochs}_cooling_factor={args.cooling_factor}',
             times=times,
             validation_accuracy=val_acc,
             validation_loss=val_loss,
             train_accuracy=train_acc,
             train_loss=train_loss,
             epochs=np.array([args.epochs]),
             cooling_factor=np.array([args.cooling_factor]),
             probabilities=probabilities,
             na=na,
             ac=ac)

