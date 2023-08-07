import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle

# Custom Libraries
import utils
import pandas as pd

from torchsummary import summary
#torch.manual_seed(1), random seeds = [1, 2, 3, 4, 5] for 5 runs

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

max_lr = 0.04
iteration_num = 0
stepsize = 10000
slope = max_lr/stepsize


delay = 1

def cal(x, beta = 5, alpha = 0.06, offset = 0.04):
    #y = (1.0/(math.exp(-k*x)+1))**a
    #y = x/(x + math.exp(c1 - c2*x))
    y = alpha*1/(1+(x/(1-x))**(-beta)) + offset

    return y
            
# Main
def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type=="reinit" else False
    print(device)
    # Data Loader
    transform=transforms.Compose([transforms.ToTensor()])
    transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet_3, densenet 

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet 

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet 

    
    # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()
    for run in range(0,10):
        data_list = list()
        data_list.append(traindataset)
        data_list.append(testdataset)
        full_dataset = torch.utils.data.ConcatDataset(data_list)

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        traindataset, testdataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        train_size = int(0.75 * len(traindataset))
        val_size = len(traindataset) - train_size
        traindataset, valdataset = torch.utils.data.random_split(traindataset, [train_size, val_size])


        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
        #train_loader = cycle(train_loader)
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

        val_loader = torch.utils.data.DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)
        
        # Importing Network Architecture
        global model
        if args.arch_type == "fc1":
           model = fc1.fc1().to(device)
        elif args.arch_type == "lenet5":
            model = LeNet5.LeNet5().to(device)
        elif args.arch_type == "alexnet":
            model = AlexNet.AlexNet().to(device)
        elif args.arch_type == "vgg19":
            model = vgg.vgg19_bn().to(device)  
        elif args.arch_type == "resnet20":
            model = resnet_3.resnet20_cifar().to(device)   
        elif args.arch_type == "densenet121":
            model = densenet.densenet121().to(device)   
        # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()

        #print(summary(model,(1,28,28)))
        # Weight Initialization
        model.apply(weight_init)
        # Copying and Saving Initial State
        initial_state_dict = copy.deepcopy(model.state_dict())
        utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
        torch.save(model, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}.pth.tar")

        # Making Initial Mask
        make_mask(model)

        # Optimizer and Loss
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

        # Layer Looper
        #for name, param in model.named_parameters():
        #    print(name, param.size())

        # Pruning
        # NOTE First Pruning Iteration is of No Compression
        bestacc = 0.0
        best_accuracy = 0
        ITERATION = args.prune_iterations
        comp = np.zeros(ITERATION,float)
        bestacc = np.zeros(ITERATION,float)
        step = 0
        all_loss = np.zeros(args.end_iter,float)
        all_accuracy = np.zeros(args.end_iter,float)


        for _ite in range(args.start_iter, ITERATION):
            global max_lr
            if _ite > delay:
                weights_pruned_per = 1 - (1 - args.prune_percent/100)**(_ite - delay)
                max_lr = cal(weights_pruned_per)
            
            global slope
            slope = max_lr/stepsize
            global iteration_num
            iteration_num = 0
            print('-----------------------------')
            print('pruning cycle '+str(_ite)+' max_lr '+str(max_lr)+' slope '+str(slope))

            if not _ite == 0:
                prune_by_percentile(args.prune_percent, _ite, model_prev, resample=resample, reinit=reinit)
                if reinit:
                    model.apply(weight_init)
                    #if args.arch_type == "fc1":
                    #    model = fc1.fc1().to(device)
                    #elif args.arch_type == "lenet5":
                    #    model = LeNet5.LeNet5().to(device)
                    #elif args.arch_type == "alexnet":
                    #    model = AlexNet.AlexNet().to(device)
                    #elif args.arch_type == "vgg16":
                    #    model = vgg.vgg16().to(device)  
                    #elif args.arch_type == "resnet18":
                    #    model = resnet.resnet18().to(device)   
                    #elif args.arch_type == "densenet121":
                    #    model = densenet.densenet121().to(device)   
                    #else:
                    #    print("\nWrong Model choice\n")
                    #    exit()
                    step = 0
                    for name, param in model.named_parameters():
                        if 'weight' in name:
                            weight_dev = param.device
                            param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                            step = step + 1
                    step = 0
                else:
                    original_initialization(mask, initial_state_dict)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
            print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

            # Print the table of Nonzeros in each layer
            comp1 = utils.print_nonzeros(model)
            comp[_ite] = comp1
            pbar = tqdm(range(args.end_iter))

            for iter_ in pbar:
                # Frequency for Testing
                if iter_ == 0:
                    curr_lr = args.lr
                #curr_lr = adjust_learning_rate(optimizer, iter_, curr_lr)
                if iter_ % args.valid_freq == 0:
                    accuracy = test(model, test_loader, criterion)
                    # Save Weights
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                        torch.save(model,f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}.pth.tar")
                # Training
                loss, curr_lr = train(model, train_loader, optimizer, criterion, curr_lr, iter_)
                all_loss[iter_] = loss
                all_accuracy[iter_] = accuracy
                train_accuracy = test(model, train_loader, criterion)

                #val_accuracy = test(model, val_loader, criterion)
                                
                if iter_ == 0:
                    record = []
                    record.append(iter_)
                    record.append(train_accuracy)
                    record.append(accuracy)
                    record.append(loss)
                else:
                    temp = []
                    temp.append(iter_)
                    temp.append(train_accuracy)
                    temp.append(accuracy)
                    temp.append(loss)
                    record = np.vstack((record,temp))
                #print(record)

                if iter_==args.end_iter-2:
                    model_prev = copy.deepcopy(model.state_dict())

                #if iter_==args.end_iter-1:
                #    num_on = cal_on(model,train_loader,args.batch_size)

                #    print('-------Overloaded Neuron Rate------' + str(num_on))

                #if _ite <= ITERATION - 1:
                #    train_accuracy = test(model, train_loader, criterion)
                #    print("epoch",iter_,"training accuracy", train_accuracy, "test accuracy", accuracy, "training loss", loss)

                # Frequency for Printing Accuracy and Loss
                if iter_ % args.print_freq == 0:
                    pbar.set_description(
                        f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       
            record = pd.DataFrame(record)
            record.to_csv('vgg_16_result_1 '+str(run)+' '+str(_ite)+'.csv',index=0, header=0)

            writer.add_scalar('Accuracy/test', best_accuracy, comp1)
            bestacc[_ite]=best_accuracy
            '''
            store_weights = []
            for name, parameter in model.named_parameters():
                if 'weight' in name:
                    weights = param.data.cpu().numpy()
                if len(store_weights) == 0:
                    store_weights = np.reshape(weights,(1,-1))
                else:
                    store_weights = np.hstack((store_weights,np.reshape(weights,(1,-1))))

            store_weights = pd.DataFrame(store_weights)
            store_weights.to_csv('original_weights.csv',index=0, header=0) 
            '''           

            # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
            #NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
            #NOTE Normalized the accuracy to [0,100] for ease of plotting.
            plt.plot(np.arange(1,(args.end_iter)+1), 100*(all_loss - np.min(all_loss))/np.ptp(all_loss).astype(float), c="blue", label="Loss") 
            plt.plot(np.arange(1,(args.end_iter)+1), all_accuracy, c="red", label="Accuracy") 
            plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})") 
            plt.xlabel("Iterations") 
            plt.ylabel("Loss and Accuracy") 
            plt.legend() 
            plt.grid(color="gray") 
            utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
            plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200) 
            plt.close()

            # Dump Plot values
            utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
            all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
            all_accuracy.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_accuracy_{comp1}.dat")
            
            # Dumping mask
            utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
            with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
                pickle.dump(mask, fp)
            
            # Making variables into 0
            best_accuracy = 0
            all_loss = np.zeros(args.end_iter,float)
            all_accuracy = np.zeros(args.end_iter,float)

        # Dumping Values for Plotting
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
        bestacc.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestaccuracy.dat")

        # Plotting
        a = np.arange(args.prune_iterations)
        plt.plot(a, bestacc, c="blue", label="Winning tickets") 
        plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
        plt.xlabel("Unpruned Weights Percentage") 
        plt.ylabel("test accuracy") 
        plt.xticks(a, comp, rotation ="vertical") 
        plt.ylim(0,100)
        plt.legend() 
        plt.grid(color="gray") 
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
        plt.close()                    
   
# Function for Training
def train(model, train_loader, optimizer, criterion, curr_lr, iter_):
    EPS = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    global iteration_num
    global slope
    
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        
        iteration_num = iteration_num + 1
        
        curr_lr = iteration_num*slope

        if iter_ >= 14:
            curr_lr = max_lr
        if iter_ >= 80:
            curr_lr = max_lr/10
        if iter_ >= 120:
            curr_lr = max_lr/100

        if iteration_num%100==0:
            print(iter_, iteration_num, curr_lr, slope)

        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
            
        #print(batch_idx)
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        #print(imgs.shape)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
    
    #for batch_idx, (imgs, targets) in enumerate(train_loader):
        # Freezing Pruned weights by making their gradients Zero
        # print(batch_idx)

        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor == EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        
        optimizer.step()
    return train_loss.item(), curr_lr


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# Prune by Percentile module
def prune_by_percentile(percent, cycle, model_prev, resample=False, reinit=False, **kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():
            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]

                prev_alive_tensor = model_prev[name].cpu().numpy()
                prev_alive = prev_alive_tensor[np.nonzero(tensor)]

                gradient = abs(alive- prev_alive)
                weight = abs(alive)

                percentile_value = np.percentile(gradient.flatten()*weight.flatten(), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor-prev_alive_tensor)*abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0
# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 0, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=160, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="reinit", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="vgg19", type=str, help="fc1 | lenet5 | alexnet | vgg19 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=25, type=int, help="Pruning iterations count")

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main(args, ITE=1)
