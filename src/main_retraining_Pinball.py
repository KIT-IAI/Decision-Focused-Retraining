




# import the necessary packages
import copy
import glob
import os
import datetime
import numpy as np
import pickle
import pandas as pd
from optimisation.ClassOpti import Opti
from modules.torch_essentials import QuantileLoss
import argparse
from sklearn.preprocessing import StandardScaler 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
from modules.torch_essentials import EarlyStopper
from torchsummary import summary
from torch.nn import ParameterList



#
# 
#

PROSUMPTION_FACTOR = 0.5

DATASETS = {
    "train" : "",
    "test" : "_2.",
    "val" : ""
}

DATASET_BOUNDS = {
        "train" : (pd.to_datetime("2010-07-08 12:00:00"), pd.to_datetime("2011-07-08")),
        "val" : (pd.to_datetime("2011-07-08 12:00:00"), pd.to_datetime("2012-06-28")),
        "test" : (pd.to_datetime("2012-07-08 12:00:00"), pd.to_datetime("2013-06-28"))
    }

RESULTS_FOLDER = "data_res/results_opti/results"
FORECAST_FOLDER = "data_res/results_forecast/results"
MODEL_SAVE_FOLDER = "models"

RESULTS_FOLDER_RETRAINING = "results"

OP_EST_NAMES = [
"model_2024_03_24_23_40_58.pt",
"model_2024_03_24_23_41_56.pt",
"model_2024_03_24_23_52_14.pt",
"model_2024_03_24_23_55_20.pt",
"model_2024_03_24_23_55_55.pt",
]


SCALER = None
SOE_STATS_SCALER = None

#
# Helper Method to get the data from the data folder
#

# get the data by building id and convert it to a dataframe
def get_data(id):

    # Get the calendar data

    calendar = {}
    for dataset in DATASETS:
        calendar[dataset] = pd.read_csv(glob.glob("data/calendar_slices/calendar_slice" + DATASETS[dataset] + ".csv")[0],parse_dates=["time"], index_col="time")

    # concat train test calendar and remove the duplicates
    calendar_merged = pd.concat([calendar["train"], calendar["test"]])
    indexes = calendar_merged.index.duplicated()
    calendar_merged = calendar_merged[~indexes]

    # iterate over every calendar collumn and shift it up to 42 hours
    list_names = []
    list_cols = []
    for collumn in calendar_merged.columns:
        for i in range(1,43):
            list_names.append("f_cal" + collumn + "_t+" + str(i))
            list_cols.append(calendar_merged[collumn].shift(-i))
    # concat the shifted collumns to the calendar dataframe
    calendar_merged = pd.concat(list_cols, axis=1)
    calendar_merged.columns = list_names
      
    # drop the rows with NaN values
    calendar_merged = calendar_merged.dropna()
    #shift the index by one hour
    calendar_merged.index = calendar_merged.index + datetime.timedelta(hours=1)    
   
    # Get historical Data

    data_ausgrid = pd.read_csv("data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv", parse_dates=["time"], index_col="time").resample("1h", closed='right').sum()[str(id)] * PROSUMPTION_FACTOR
    
    # get the ground truth data
    ground_truth = {}
    for dataset in DATASETS:
        ground_truth[dataset] = {}
        ground_truth[dataset] = pd.read_csv(glob.glob("data/"+ FORECAST_FOLDER +"/"+ str(id)+ "/*/gt_forecast" + DATASETS[dataset]  + ".csv")[0], index_col=0,parse_dates=True)[DATASET_BOUNDS[dataset][0]:DATASET_BOUNDS[dataset][1]]

    #slice the historical data according to the ground truth data get the 168 values from index before the first ground truth value
    data_ausgrid = data_ausgrid[ground_truth["train"].index[0] - datetime.timedelta(hours=168):]
    # get 168 historical values in the collumns corresponding to the ground truth data at the same index    
    tmp_data = pd.DataFrame()
    list_names = []
    list_cols = []

    for i in range(0,168):
        list_names.append("f_hist_t-"+str(i))
        list_cols.append(data_ausgrid.shift(i))

    # drop the rows with NaN values 
    tmp_data = pd.concat(list_cols, axis=1)
    tmp_data.columns = list_names
    
    data_ausgrid = tmp_data.dropna()
    # increase the index by one hour
    data_ausgrid.index = data_ausgrid.index + datetime.timedelta(hours=1)
    # get the historical data for the training set

    # get basic statistical information about the building

    min = ground_truth["train"].min().values[0]
    max = ground_truth["train"].max().values[0]
    mean = ground_truth["train"].mean().values[0]
    std = ground_truth["train"].std().values[0]


    # create a dataframe with the statistical information
    statistical_information = pd.DataFrame({"min": min, "max": max, "mean": mean, "std": std}, index=[0])



    train = ground_truth["train"].join(data_ausgrid,rsuffix='hist').join(calendar_merged,rsuffix='cal')
    val = ground_truth["val"].join(data_ausgrid,rsuffix='hist').join(calendar_merged,rsuffix='cal')
    test = ground_truth["test"].join(data_ausgrid,rsuffix='hist').join(calendar_merged,rsuffix='cal')

    # get the SoE of Benchi

    # Adjust this according to you forecast you want to use in retraining
    benchi = pickle.load(open("data/"+ RESULTS_FOLDER +"/results_benchmark/" + str(id) + "/result_FC_with_Cov_MAE" , "rb"))
    benchi_2 = pickle.load(open("data/"+ RESULTS_FOLDER +"/results_benchmark/" + str(id) + "/result_FC_with_Cov_MAE_2." , "rb"))

    soe_benchi = benchi.results_online_final["e"]["house0"][0]
    soe_benchi_2 = benchi_2.results_online_final["e"]["house0"][0]

    soe_train = soe_benchi[DATASET_BOUNDS["train"][0]:DATASET_BOUNDS["train"][1]]
    soe_val = soe_benchi[DATASET_BOUNDS["val"][0]:DATASET_BOUNDS["val"][1]]
    soe_test = soe_benchi_2[DATASET_BOUNDS["test"][0]:DATASET_BOUNDS["test"][1]]


    # returning a dictionary with the data
    data = { "train" : train, "val" : val, "test" : test, "soe_train" : soe_train , "soe_val" :soe_val, "soe_test" :soe_test,  "statistical_information" : statistical_information }

    return data

#
# Dataset Class for the OpNet
#


class RetrainingDataset(Dataset):
    def __init__(self, building_id, dataset):
        data = get_data(building_id)
        self.soe = data[f"soe_{dataset}"].values
        self.stats = data["statistical_information"].values
        data_dataset = data[dataset]
        self.soe_stats = None

        self.features = data_dataset.iloc[:,data_dataset.columns.str.startswith("f_")]
        self.value = data_dataset[[str(i) for i in range(0,42)]]
        

        # scale the data
        global SCALER
        if SCALER == None:
            SCALER = StandardScaler()
            SCALER.fit(self.features)
        self.features = SCALER.transform(self.features)


        # convert the data to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.value = torch.tensor(self.value.values, dtype=torch.float32)


        # scale the SOE and the STATS
        global SOE_STATS_SCALER
        if SOE_STATS_SCALER == None:
            # load the scaler
            f = open("models/feature_scaler/feature_scaler.pkl", 'rb')
            SOE_STATS_SCALER = pickle.load(f)

        #transform
    
        self.stats = np.array([np.full((len(self.soe)),x) for x in self.stats[0]])

        self.soe_stats = SOE_STATS_SCALER.transform(np.concatenate((self.soe.reshape((len(self.soe),1)), self.stats.T),axis=1))

        self.soe_stats = torch.tensor(self.soe_stats, dtype=torch.float32)  



    # return the length of the dataset
    def __len__(self):
        return len(self.features)
    # return the features and the value
    def __getitem__(self, idx):
        return self.features[idx], self.value[idx] , self.soe_stats[idx]
    # return the shape of the data
    def data_shape(self):
        return "Features: ", self.features.shape, "Value: ", self.value.shape , "SoE_Stats: ", self.soe_stats.shape

#
# Simple Neural Network using Pytorch used later as Op
#

class FCEstimator(nn.Module):

    def __init__(self , input_size , output_size):
        super(FCEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, input):

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
class OptiEstimator(nn.Module):

    def __init__(self , output_size):
        super(OptiEstimator, self).__init__()
        self.fc0_1 = nn.Linear(42, 64)
        self.fc0_2 = nn.Linear(42, 64)


        self.fc1 = nn.Linear(133, 64) 
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, output_size)

    def forward(self, input):
        # split the first 42 from the input
        x_0 = input[:,:42]
        # split the second 42 from the input
        x_1 = input[:,42:84]
        #split the last values from the input
        x_2 = input[:,84:]

        x_0 = F.selu(self.fc0_1(x_0))
        x_1 = F.selu(self.fc0_2(x_1))


        concat = torch.cat((x_0,x_1,x_2),1)

        x = F.selu(self.fc1(concat))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))

        return x

    
# build a model containing the FCEstimator and the OptiEstimator
class RetrainingModel(nn.Module):

    def __init__(self , OptiEstimators, FCEstimator):
        super(RetrainingModel, self).__init__()
        self.OptiEstimators = OptiEstimators
        self.FCEstimator = FCEstimator


    def forward(self, input, y_batch , exog_batch):

        fc = self.FCEstimator(input)
        op_est = []
        input_op = torch.cat((fc, y_batch, exog_batch),1)
        for x in self.OptiEstimators:
            
            op_est.append(x(input_op))

        op_est = torch.stack(op_est).mean(dim=0)

        return fc, op_est
    
    def get_fc(self, input):
        return self.FCEstimator(input)


#
# Weight Adjustment method
#

def adjust_weights(loss_forecast , loss_opt_est):
        sum_loss = abs(loss_forecast) + abs(loss_opt_est)
        weight_fc =  1 - (abs(loss_forecast) / sum_loss)
        weight_op_est =  1 - (abs(loss_opt_est) / sum_loss)
        return weight_fc, weight_op_est


#
# Main Method
#   



   
if __name__ == "__main__":



    
    #
    # Argument Parser for a List of building ids used for training  
    #
    parser = argparse.ArgumentParser(description='Get the training information needed')
    parser.add_argument('-id', type=int,  default=109)


    # argument for the name regarding the wohle run
    parser.add_argument('-run_name', type=str,  default="test")

    parser.add_argument('-quantile', type=float,  default=0.5)
    args = parser.parse_args()


    # make directory for the results
    os.makedirs(RESULTS_FOLDER_RETRAINING + "/"+ args.run_name + "/" + str(args.id))



    config={
    "learning_rate": 0.0001,
    "architecture": "MLP",
    "train_building": args.id,
    "test_building": args.id,
    "val_building": args.id,
    "epochs": 1000,
    "Datasets": DATASETS,
    "Dataset_bounds": DATASET_BOUNDS,
    "Results_folder": RESULTS_FOLDER,
    "Forecast_folder": FORECAST_FOLDER,
    "Model_save_folder": MODEL_SAVE_FOLDER,
    "Op_est_name": OP_EST_NAMES,
    "Results_folder_retraining": RESULTS_FOLDER_RETRAINING,
    "Run_name": args.run_name,
    }




    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="Op-Net-Retraining",
    
    # track hyperparameters and run metadata

    config=config,
    
    )

    print("Preparing Data")
    print("Building Data for Buildings: ", args.id)

    training_data = RetrainingDataset(wandb.config["train_building"], "train")
    print("Training Data :" , training_data.data_shape())

    validation_data = RetrainingDataset(wandb.config["val_building"], "val")
    print("Validation Data :" , validation_data.data_shape())

    test_data = RetrainingDataset(wandb.config["test_building"], "test")
    print("Test Data :" , test_data.data_shape())

    print("Start Training")
    # define a simple neural network using pytorch  
    network = FCEstimator(training_data.features.shape[1], training_data.value.shape[1])
    # define the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=wandb.config["learning_rate"])
    # define the loss function
    loss_function = QuantileLoss([args.quantile])

    # define the batch size
    batch_size = 32
    
    # define the number of epochs   
    epochs = wandb.config["epochs"]

    # define the training data loader   
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True
                                                , num_workers=22)
    # define the validation data loader
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data), shuffle=False      
                                                , num_workers=22)
    # define the test data loader
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False
                                                , num_workers=22)
    
    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)
    
    network.train()
    network.to("cuda")

    # Load the opnet model
    op_models = []
    for name in wandb.config["Op_est_name"]:

        op_est_model = OptiEstimator(1) # initialize your model class
        op_est_model.load_state_dict(torch.load(MODEL_SAVE_FOLDER + "/models_optimisation/with_SoE/" +  name))
        op_est_model.to("cuda")
        op_models.append(op_est_model)
    



    val_losses_epoch = [] 
    # define the training loop
    for epoch in range(epochs):
        network.train() 
        for i, data in enumerate(train_loader, 0):
            losses = []
            # get the inputs
            x_batch, y_batch , exog_batch = data

            x_batch = x_batch.to("cuda")    
            y_batch = y_batch.to("cuda")    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(x_batch)
            loss = loss_function(outputs, y_batch)

            # backpropagate the loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())


        # print statistics
        print("Epoch: ", epoch, " Loss: ", np.asarray(losses).mean())

        val_losses = []
        # define the validation loop
        network.eval()
        for i, data in enumerate(val_loader, 0):
            val_losses = []
            # get the inputs
            x_batch, y_batch ,exog_batch = data
            x_batch = x_batch.to("cuda")    
            y_batch = y_batch.to("cuda")    
            with torch.no_grad():
                outputs = network(x_batch)
                loss = loss_function(outputs, y_batch)
                val_losses.append(loss.item())
            
        epoch_val_loss = np.asarray(val_losses).mean()
        print("Validation Loss: ", epoch_val_loss)
        val_losses_epoch.append(epoch_val_loss)


        # log the loss and accuracy values at the end of each epoch
        wandb.log({"epoch": epoch, "loss": np.asarray(losses).mean(), "val_loss": np.asarray(val_losses).mean()})

        # if val loss is the minimum val loss so far save the model local
        if epoch_val_loss == np.asarray(val_losses_epoch).min():
            model = copy.deepcopy(network)
        
        if early_stopper.early_stop(epoch_val_loss): 
            print("Early Stopping, restoring the best weights")
            network = model            
            break
            
    print("Finished Training")

    network = model

    # define the test
    test_loss_MSE = []
    test_loss_MAE = []
    test_loss_op_est = []
    test_loss_used = []

    network.eval()
    for i, data in enumerate(test_loader, 0):
        # get the inputs

        x_batch, y_batch, exog_batch = data
        x_batch = x_batch.to("cuda")    
        y_batch = y_batch.to("cuda")    
        
        with torch.no_grad():
            outputs = network(x_batch)
            outputs = outputs.cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            loss_function_MSE = nn.MSELoss()
            loss_function_MAE = nn.L1Loss()
            loss_MSE = loss_function_MSE(torch.tensor(outputs), torch.tensor(y_batch))
            loss_MAE = loss_function_MAE(torch.tensor(outputs), torch.tensor(y_batch))
            used_loss = loss_function(torch.tensor(outputs), torch.tensor(y_batch))
            
            losses_op_est_tmp = []
            for op_est_model in op_models:


                losses_op_est_tmp.append(torch.sum(op_est_model(torch.cat((torch.tensor(outputs).to("cuda"), torch.tensor(y_batch).to("cuda"), torch.tensor(torch.reshape(exog_batch, (355,5))).to("cuda")),1))).item())
            
            test_loss_op_est.append(np.array(losses_op_est_tmp).mean())
            test_loss_MSE.append(loss_MSE.item())
            test_loss_MAE.append(loss_MAE.item())
            test_loss_used.append(used_loss.item())

            

            save_results = pd.DataFrame(outputs, columns=[str(i) for i in range(0,42)])
            #get the index
            save_results["time"] = pd.date_range(start=DATASET_BOUNDS["test"][0], end=DATASET_BOUNDS["test"][1], freq="1d")
            save_results = save_results.set_index("time")

            #save save_results
            save_results.to_csv(RESULTS_FOLDER_RETRAINING + "/"+ args.run_name + "/" + str(args.id) + "/" +  "fc.csv")

            save_results = pd.DataFrame(y_batch, columns=[str(i) for i in range(0,42)])
            #get the index
            save_results["time"] = pd.date_range(start=DATASET_BOUNDS["test"][0], end=DATASET_BOUNDS["test"][1], freq="1d")
            save_results = save_results.set_index("time")

            #save save_results
            save_results.to_csv(RESULTS_FOLDER_RETRAINING + "/"+ args.run_name + "/" + str(args.id) + "/" + "/" +  "gt.csv")

        


    
    print("Before retraining Test Loss MSE: ", np.asarray(test_loss_MSE).mean())
    print("Before retraining Test Loss MAE: ", np.asarray(test_loss_MAE).mean())
    print("Before retraining Test Loss Op Est: ", np.asarray(test_loss_op_est).mean())
    print("Before retraining Test Used Loss: ", np.asarray(test_loss_used).mean())


    # log the loss and accuracy values at the end 
    wandb.log({"test_loss_MSE_br": np.asarray(test_loss_MSE).mean(), "test_loss_MAE_br": np.asarray(test_loss_MAE).mean(), "loss_op_est_br": np.asarray(test_loss_op_est).mean(), "loss_used_br": np.asarray(test_loss_used).mean()})
    
    # save the model with time stamp in a models folder which it creates if it does not exist
    name = "model_" + str(datetime.datetime.now()) + ".pt"

    os.makedirs(MODEL_SAVE_FOLDER + "/models_before_retraining/" + wandb.config["Run_name"], exist_ok=True)
    
    torch.save(network.state_dict(), MODEL_SAVE_FOLDER + "/models_before_retraining/" + wandb.config["Run_name"] + "/" + name)
    wandb.log({"network_before_re_saved_as": name})
    

    # save the model in the wandb cloud storage
    torch.save(network.state_dict(), wandb.run.dir + "/model.pt")
    



    #
    #
    #
    #
    #
    # start the retraining process
    #
    #
    #
    #

    #inialize the big network


    network_retrain = RetrainingModel(ParameterList(op_models), copy.deepcopy(network))
    network_retrain.to("cuda")

    for name, param in network_retrain.named_parameters():
        if name.startswith("OptiEstimator"):
            param.requires_grad = False
    
    summary(network_retrain, [(training_data.features.shape[1],), (42,), (5,)])
    # define the optimizer
    optimizer = torch.optim.Adam(network_retrain.parameters(), lr=wandb.config["learning_rate"])
    # define the loss function
    loss_function = QuantileLoss([args.quantile])

    # define the batch size
    batch_size = 32

    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)

    weight_op_est = 0
    weight_fc = 0

    val_losses_epoch = []
    val_losses_fc_epoch = []
    val_losses_op_est_epoch = []

    # define the training loop
    for epoch in range(epochs):
        network_retrain.train()
        for i, data in enumerate(train_loader, 0):
            losses = []
            losses_fc = []
            losses_op_est = []
            # get the inputs
            x_batch, y_batch , exog_batch = data

            x_batch = x_batch.to("cuda")    
            y_batch = y_batch.to("cuda")  
            exog_batch = exog_batch.to("cuda")  
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            fc, op_est  = network_retrain(x_batch,y_batch,exog_batch)

            #
            # Calc batch losses
            #
            loss_fc = loss_function(fc, y_batch)
            loss_op_est = torch.mean(op_est)
            weight_fc, weight_op_est = adjust_weights(loss_forecast = np.asarray(loss_fc.detach().cpu()) , loss_opt_est = np.asarray(loss_op_est.detach().cpu()))
            loss = weight_fc * loss_fc + weight_op_est * loss_op_est

            losses_fc.append(loss_fc.item())
            losses_op_est.append(loss_op_est.item())
            losses.append(loss.item())


            # backpropagate the loss
            loss.backward()
            optimizer.step()


        # print statistics
        print("Epoch: ", epoch, " Loss: ", np.asarray(losses).mean() , " Loss Forecast: ", np.asarray(losses_fc).mean() , " Loss Op Est: ", np.asarray(losses_op_est).mean())
        network_retrain.eval()
        val_losses = []
        val_losses_fc = []
        val_losses_op_est = []
        # define the validation loop
        for i, data in enumerate(val_loader, 0):

            # get the inputs
            x_batch, y_batch , exog_batch = data
            x_batch = x_batch.to("cuda")    
            y_batch = y_batch.to("cuda")    
            exog_batch = exog_batch.to("cuda")  
            with torch.no_grad():
                fc, op_est = network_retrain(x_batch,y_batch,exog_batch)
                val_loss_fc = loss_function(fc, y_batch)
                val_loss_op_est = torch.mean(op_est)
                weight_fc, weight_op_est = adjust_weights(loss_forecast = np.asarray(val_loss_fc.cpu()) , loss_opt_est = np.asarray(val_loss_op_est.cpu()))
                val_loss = weight_fc * val_loss_fc + weight_op_est * val_loss_op_est

                val_losses.append(val_loss.item())
                val_losses_fc.append(val_loss_fc.item())
                val_losses_op_est.append(val_loss_op_est.item())

        epoch_val_loss = np.asarray(val_losses).mean()
        epoch_val_loss_fc = np.asarray(val_losses_fc).mean()    
        epoch_val_loss_op_est = np.asarray(val_losses_op_est).mean()

        print("Validation Loss: ", epoch_val_loss)
        val_losses_epoch.append(epoch_val_loss)
        val_losses_fc_epoch.append(epoch_val_loss_fc)
        val_losses_op_est_epoch.append(epoch_val_loss_op_est)

        # log the loss and accuracy values at the end of each epoch
        wandb.log({"epoch_retrain": epoch, "loss_retrain": np.asarray(losses).mean(), "loss_fc_retrain": np.asarray(losses_fc).mean(), "loss_op_est_retrain": np.asarray(losses_op_est).mean() ,"val_loss_retrain": epoch_val_loss, "val_loss_fc_retrain": epoch_val_loss_fc, "val_loss_op_est_retrain": epoch_val_loss_op_est})

        # if val loss is the minimum val loss so far save the model local
        if epoch_val_loss == np.asarray(val_losses_epoch).min() or epoch == 0:
            print("Saving the model")
            tmp_model = copy.deepcopy(network_retrain)
        
        if early_stopper.early_stop(epoch_val_loss): 
            print("Early Stopping, restoring the best weights")
            network_retrain = tmp_model            
            break
            
    print("Finished Training")


    # define the test
    test_loss_MSE = []
    test_loss_MAE = []
    test_loss_op_est = []
    test_loss_used = []
    network_retrain.eval()
    for i, data in enumerate(test_loader, 0):
        # get the inputs

        x_batch, y_batch , exog_batch = data
        x_batch = x_batch.to("cuda")    
        y_batch = y_batch.to("cuda")  
        exog_batch = exog_batch.to("cuda")    
        
        with torch.no_grad():
            outputs = network_retrain.get_fc(x_batch)
            outputs = outputs.cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            loss_function_MSE = nn.MSELoss()
            loss_function_MAE = nn.L1Loss()
            loss_MSE = loss_function_MSE(torch.tensor(outputs), torch.tensor(y_batch))
            loss_MAE = loss_function_MAE(torch.tensor(outputs), torch.tensor(y_batch))
            used_loss = loss_function(torch.tensor(outputs), torch.tensor(y_batch))

            losses_op_est_tmp = []
            for op_est_model in op_models:
                losses_op_est_tmp.append(torch.sum(op_est_model(torch.cat((torch.tensor(outputs).to("cuda"), torch.tensor(y_batch).to("cuda"), torch.tensor(torch.reshape(exog_batch, (355,5))).to("cuda")),1))).item())
            

            test_loss_op_est.append(np.array(losses_op_est_tmp).mean())

            test_loss_MSE.append(loss_MSE.item())
            test_loss_MAE.append(loss_MAE.item())
            test_loss_used.append(used_loss.item())

            save_results = pd.DataFrame(outputs, columns=[str(i) for i in range(0,42)])
            #get the index
            save_results["time"] = pd.date_range(start=DATASET_BOUNDS["test"][0], end=DATASET_BOUNDS["test"][1], freq="1d")
            save_results = save_results.set_index("time")
            #save save_results
            save_results.to_csv(RESULTS_FOLDER_RETRAINING + "/"+ str(args.run_name) + "/" + str(args.id) + "/" +  "fc_retraining.csv")

    print("After retraining Test Loss MSE: ", np.asarray(test_loss_MSE).mean())
    print("After retraining Test Loss MAE: ", np.asarray(test_loss_MAE).mean())
    print("After retraining Test Loss Op Est: ", np.asarray(test_loss_op_est).mean())
    print("After retraining Test Used Loss: ", np.asarray(test_loss_used).mean())

    # log the loss and accuracy values at the end 
    wandb.log({"test_loss_MSE_ar": np.asarray(test_loss_MSE).mean(), "test_loss_MAE_ar": np.asarray(test_loss_MAE).mean(), "loss_op_est_ar": np.asarray(test_loss_op_est).mean(), "loss_used_ar": np.asarray(test_loss_used).mean()})


    fc_model = network_retrain.FCEstimator
    fc_model.to("cpu")

    name = "model_" + str(datetime.datetime.now()) + ".pt"
    
    
    os.makedirs(MODEL_SAVE_FOLDER + "/models_after_retraining/" + wandb.config["Run_name"], exist_ok=True)

    torch.save(fc_model.state_dict(), MODEL_SAVE_FOLDER + "/models_after_retraining/" + wandb.config["Run_name"] + "/" + name)

    wandb.log({"network_after_re_saved_as": name})
    
    torch.save(fc_model.state_dict(), wandb.run.dir + "/model_re.pt")

# mark the run as finished    
wandb.finish()


    



   