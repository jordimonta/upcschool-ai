import torch
import ray
from ray import tune
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

    ####### tune ############
    tune.report(mean_accuracy=accuracy_avg)
    
    return {'Eval epoch accuracy ' : accuracy_avg}    

def train_model(config,train_dataset,val_dataset):
        
    my_model = MyModel(config).to(device)    

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batchsize"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batchsize"], shuffle=False)

    for epoch in range(int(config["epochs"])):
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

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])
    my_dataset = MyDataset("/home/manager/upcschool-ai/data/ChineseMNIST/",
                        "/home/manager/upcschool-ai/data/ChineseMNIST/chinese_mnist.csv",
                        transform=trans)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset,[10000, 2500, 2500])    

    ray.init(configure_logging=False)
    analysis = tune.run(
        tune.with_parameters(train_model, train_dataset=train_dataset,val_dataset=val_dataset),
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.99
        },        
        num_samples=5,
        config={
            "epochs": tune.uniform(1, 10),
            "lrate": 0.001,
            "batchsize": 100,
            "activation": tune.grid_search(["relu", "tanh"]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)

#RESULTS
# Number of trials: 10/10 (10 TERMINATED)
# +--------------------+------------+-------+--------------+----------+--------+--------+------------------+
# | Trial name         | status     | loc   | activation   |   epochs |    acc |   iter |   total time (s) |
# |--------------------+------------+-------+--------------+----------+--------+--------+------------------|
# | _inner_d7cd1_00000 | TERMINATED |       | relu         |  8.14001 | 0.8484 |      8 |         3146.66  |
# | _inner_d7cd1_00001 | TERMINATED |       | tanh         |  6.79278 | 0.8748 |      6 |         2415.06  |
# | _inner_d7cd1_00002 | TERMINATED |       | relu         |  7.40052 | 0.3904 |      7 |         2785.51  |
# | _inner_d7cd1_00003 | TERMINATED |       | tanh         |  5.37681 | 0.8432 |      5 |         2041.56  |
# | _inner_d7cd1_00004 | TERMINATED |       | relu         |  4.06012 | 0.252  |      4 |         1504.87  |
# | _inner_d7cd1_00005 | TERMINATED |       | tanh         |  7.59784 | 0.9288 |      7 |         2635.5   |
# | _inner_d7cd1_00006 | TERMINATED |       | relu         |  6.28263 | 0.3148 |      6 |         2251.97  |
# | _inner_d7cd1_00007 | TERMINATED |       | tanh         |  6.34101 | 0.8664 |      6 |         2175.55  |
# | _inner_d7cd1_00008 | TERMINATED |       | relu         |  5.31423 | 0.7048 |      5 |         1794.67  |
# | _inner_d7cd1_00009 | TERMINATED |       | tanh         |  4.30082 | 0.8444 |      4 |          866.324 |
# +--------------------+------------+-------+--------------+----------+--------+--------+------------------+


# Best hyperparameters found were:  {'epochs': 7.59783548805109, 'lrate': 0.001, 'batchsize': 100, 'activation': 'tanh'}



    # Put de best configuration here or execute main.py with the best configuration
    # config = {
    #     "epochs": 8,
    #     "lrate": 0.001,
    #     "batchsize": 100,
    #     "activation": "tanh"
    # }
    # model_trained=train_model(config,train_dataset,val_dataset)

    # save_model(model_trained, "/home/manager/upcschool-ai/upcschool-ai/session-2/model.best.save")

    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batchsize"], shuffle=False)
    # print(test_model(model_trained,
    #                 torch.nn.CrossEntropyLoss(),
    #                 test_dataloader)
    #     )


