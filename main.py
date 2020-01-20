import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

plt.rcParams["figure.figsize"] = (20,10)
sns.set()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 5e-3
epoch = 10
batch_size = 128

data_transforms = transforms.Compose([
                      transforms.RandomResizedCrop(128),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])

train_datas = ImageFolder(root='/content/gdrive/My Drive/EIB/datas/train/', transform=data_transforms)
test_datas = ImageFolder(root='/content/gdrive/My Drive/EIB/datas/test/', transform=data_transforms)

train_dataloader = torch.utils.data.DataLoader(train_datas, batch_size=batch_size, shuffle=True, num_workers=64)
test_dataloader = torch.utils.data.DataLoader(test_datas, batch_size=batch_size, shuffle=True, num_workers=64)

class_names = train_datas.classes

data_iter = iter(train_datas)

NB_TRY = 50

NB_N = 10
NB_K = 10
NB_P = 10

models = []

accuracies_train = []
accuracies_val = []

N = [i for i in range(5,5+NB_N)]
N_proba = [1/NB_N for i in range(5,5+NB_N)]
N_He = [0 for i in range(NB_N)]
N_Ha = [0.0 for i in range(NB_N)]


K = [i for i in range(1,1+NB_K)]
K_proba = [1/NB_K for i in range(1,1+NB_K)]
K_He = [0 for i in range(NB_K)]
K_Ha = [0.0 for i in range(NB_K)]

P = [i/NB_P for i in range(NB_P)]
P_proba = [1/NB_P for i in range(NB_P)]
P_He = [0 for i in range(NB_P)]
P_Ha = [0.0 for i in range(NB_P)]


print("\033[4mRésumé de la recherche d'architechture :\033[0m")
print("\t-Nombre d'essaies : ", NB_TRY)
print("\t-Tailles N possibles : ", N)
print("\t-Connexions K possibles : ", K)
print("\t-Probabilités P possibles : ", P)
for i in range(NB_TRY):
    

    N_temp_idx = list(np.random.multinomial(1, N_proba, size=1)[0]).index(1)
    N_temp = N[N_temp_idx]

    K_temp_idx = list(np.random.multinomial(1, K_proba, size=1)[0]).index(1)
    K_temp = K[K_temp_idx]

    while K_temp > N_temp - 2:
        K_temp_idx = list(np.random.multinomial(1, K_proba, size=1)[0]).index(1)
        K_temp = K[K_temp_idx]


    P_temp_idx = list(np.random.multinomial(1, P_proba, size=1)[0]).index(1)
    P_temp = P[P_temp_idx]

    
    print("\n\033[4mEssai n°{}\033[0m".format(i))
    print("\t", N_temp)
    print("\t", K_temp)
    print("\t", P_temp)


    try:
        print("\t-Probabilités N : ", N_proba)
        print("\t-Probabilités K : ", K_proba)
        print("\t-Probabilités P : ", P_proba)
        model = ModelNAS(N = N_temp, K = K_temp, P = P_temp, number_layer = 5, number_class=2).cuda(0)
        accuracy_train, accuracy_val = train(model, train_dataloader, test_dataloader, lr)
        models.append(model)
        accuracies_train.append(accuracy_train)
        accuracies_val.append(accuracy_val)

        
        N_He[N_temp_idx] +=1
        K_He[K_temp_idx] +=1
        P_He[K_temp_idx] +=1
        
        N_Ha[N_temp_idx] = accuracy_val.item()
        K_Ha[K_temp_idx] = accuracy_val.item()
        P_Ha[P_temp_idx] = accuracy_val.item()

        N_proba = DeltaHUpdate(N_Ha, N_He, N_proba)
        K_proba = DeltaHUpdate(K_Ha, K_He, K_proba)
        P_proba = DeltaHUpdate(P_Ha, P_He, P_proba)

        print("\t Nombre de paramètres : ", sum(p.numel() for p in model.parameters()))
        print()
        print("\t\tRéussi !")
    except:
        print("\t\tRâté !")

plt.plot(N, N_proba)
plt.plot(K, K_proba)
plt.show()

plt.plot(P, P_proba)
plt.show()