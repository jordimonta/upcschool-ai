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
# | _inner_ddf78_00000 | TERMINATED |       | relu         |  3.28255 | 0.9236 |      3 |         1146.1   |
# | _inner_ddf78_00001 | TERMINATED |       | tanh         |  5.5406  | 0.9468 |      5 |         1934.41  |
# | _inner_ddf78_00002 | TERMINATED |       | relu         |  9.6181  | 0.948  |      9 |         3444.15  |
# | _inner_ddf78_00003 | TERMINATED |       | tanh         |  1.62362 | 0.8408 |      1 |          390.541 |
# | _inner_ddf78_00004 | TERMINATED |       | relu         |  7.35741 | 0.952  |      7 |         2670.29  |
# | _inner_ddf78_00005 | TERMINATED |       | tanh         |  4.83323 | 0.946  |      4 |         1540.81  |
# | _inner_ddf78_00006 | TERMINATED |       | relu         |  5.26032 | 0.9504 |      5 |         1905.59  |
# | _inner_ddf78_00007 | TERMINATED |       | tanh         |  9.82794 | 0.9568 |      9 |         2396.35  |
# | _inner_ddf78_00008 | TERMINATED |       | relu         |  3.43788 | 0.9068 |      3 |         1067.85  |
# | _inner_ddf78_00009 | TERMINATED |       | tanh         |  3.87895 | 0.9224 |      3 |          913.847 |
# +--------------------+------------+-------+--------------+----------+--------+--------+------------------+


# Best hyperparameters found were:  {'epochs': 9.827942822722006, 'lrate': 0.001, 'batchsize': 100, 'activation': 'tanh'}



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


