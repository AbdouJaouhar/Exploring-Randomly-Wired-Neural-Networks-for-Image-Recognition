import numpy as np
import torch
import torch.nn as nn
 
from scipy.special import softmax


def train(model, train_dataloader, test_dataloader, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    epoch = 10
    dataloaders = {'train' : train_dataloader, 'val' : test_dataloader}

    accuracy = {'train' : 0.0, 'val' : 0.0}

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        with tqdm(total=len(dataloaders[phase])) as pbar:
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.update(1)
                pbar.set_description('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss/((i+1)*batch_size), running_corrects.double()/((i+1)*batch_size)))


        accuracy[phase] = running_corrects.double() / ((i+1)*batch_size)
    
    return accuracy['train'], accuracy['val']


def DeltaHUpdate(Ha, He, p, alpha=0.5):
    deltaHa = np.array([h-np.array(Ha) for h in Ha])
    deltaHe = np.array([h-np.array(He) for h in He])
    # print("DHa = ", np.round(deltaHa,2))
    # print("DHe = ", np.round(deltaHe,2))
    for i in range(len(Ha)):
        sum_upd = 0
        for j in range(len(Ha)):
            if deltaHe[i,j] < 0 and deltaHa[i,j] > 0:
                ce =1
            else:
                ce = 0

            if deltaHe[i,j] > 0 and deltaHa[i,j] < 0:
                ca =1
            else:
                ca = 0
            
            sum_upd += ce-ca

        p[i] = p[i] + alpha*sum_upd
    return softmax(p)