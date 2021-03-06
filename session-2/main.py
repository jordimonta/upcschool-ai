import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    loss_function: torch.nn.Module, 
                    data_loader: torch.utils.data.DataLoader):
    # switch to train mode
    model.train()

    for data in data_loader:        
        X, y = data
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        output = model(X) #.view(-1, 4096)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

def eval_single_epoch(model: torch.nn.Module,
                    loss_function: torch.nn.Module, 
                    data_loader: torch.utils.data.DataLoader):
    # switch to evaluate mode                
    model.eval()

    accuracy_total = 0
    
    for data in data_loader:
        X, y = data
        X, y = X.to(device), y.to(device)
        output = model(X) #.view(-1, 4096)
        loss = loss_function(output, y)
        accuracy_total += accuracy(y,output)
        
    accuracy_avg = 100.0 * accuracy_total / len(data_loader.dataset)
    
    return {'Eval epoch accuracy ' : accuracy_avg}    

def train_model(config,train_dataset,val_dataset):
        
    my_model = MyModel(config).to(device)    

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batchsize"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batchsize"], shuffle=False)

    for epoch in range(config["epochs"]):
        train_single_epoch(my_model,
                        torch.optim.Adam(my_model.parameters(), config["lrate"]),
                        torch.nn.CrossEntropyLoss(),
                        train_dataloader)
        print(eval_single_epoch(my_model,
                        torch.nn.CrossEntropyLoss(),
                        val_dataloader)
            )

    return my_model

def test_model(model: torch.nn.Module,
            loss_function: torch.nn.Module, 
            data_loader: torch.utils.data.DataLoader):
    model.eval

    accuracy_total = 0
    
    for data in data_loader:
        X, y = data
        X, y = X.to(device), y.to(device)
        output = model(X)#.view(-1, 4096)
        loss = loss_function(output, y)
        accuracy_total += accuracy(y,output)
        
    accuracy_avg = 100.0 * accuracy_total / len(data_loader.dataset)
    
    return {'Test accuracy' : accuracy_avg}    
        

if __name__ == "__main__":

    config = {
        "epochs": 10,
        "lrate": 0.001,
        "batchsize": 100,
        "activation": "tanh"
    }
    
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])
    my_dataset = MyDataset("/home/manager/upcschool-ai/data/ChineseMNIST/",
                        "/home/manager/upcschool-ai/data/ChineseMNIST/chinese_mnist.csv",
                        transform=trans)

    dataset_size = len(my_dataset)
    print(dataset_size)
    #train_size = int(0.6 * dataset_size)
    #test_size = int(0.2 * dataset_size)
    #validate_size = dataset_size - train_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset,[10000, 2500, 2500])    

    model_trained=train_model(config,train_dataset,val_dataset)

    save_model(model_trained, "/home/manager/upcschool-ai/upcschool-ai/session-2/model_best2.save")

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batchsize"], shuffle=False)
    print(test_model(model_trained,
                    torch.nn.CrossEntropyLoss(),
                    test_dataloader)
        )

#### RESULTS 8 epochs #####
# 'Eval epoch accuracy ': 0.864399995803833}
# {'Eval epoch accuracy ': 0.9264000010490417}
# {'Eval epoch accuracy ': 0.9112000012397766}
# {'Eval epoch accuracy ': 0.9456000113487244}
# {'Eval epoch accuracy ': 0.9608000063896179}
# {'Eval epoch accuracy ': 0.9508000087738037}
# {'Eval epoch accuracy ': 0.9624000096321106}
# {'Eval epoch accuracy ': 0.9660000014305115}
# {'Test accuracy': 0.970400002002716}

#### RESULTS 10 epochs #####
# {'Eval epoch accuracy ': 0.8627999997138978}
# {'Eval epoch accuracy ': 0.9203999996185303}
# {'Eval epoch accuracy ': 0.9432000041007995}
# {'Eval epoch accuracy ': 0.9508000040054321}
# {'Eval epoch accuracy ': 0.9456000018119812}
# {'Eval epoch accuracy ': 0.9564000034332275}
# {'Eval epoch accuracy ': 0.9676000022888184}
# {'Eval epoch accuracy ': 0.9692000031471253}
# {'Eval epoch accuracy ': 0.9684000086784362}
# {'Eval epoch accuracy ': 0.9720000076293945}
# {'Test accuracy': 0.9756000089645386}