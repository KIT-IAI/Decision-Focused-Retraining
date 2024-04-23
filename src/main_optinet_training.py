# Write a main method for the Optinet class that reads in the data from the data folder which are given in pickel format
# afterwards use the data to train an neural network.




# import the necessary packages
import glob
import datetime
import copy
import numpy as np
import pickle
import pandas as pd
from optimisation.ClassOpti import Opti
import argparse
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler , QuantileTransformer as SklearnQuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
from modules.torch_essentials import EarlyStopper, l1_regularizer, l2_regularizer
import plotly.express as px



#
# 
#

IMBALANCE_FACTOR = 10


FORECASTING_METHOD = [
    ""
]

LOSS_FUNCTION = [
    "Huber",
    "MSE",
    "MAE",
    "Pinball_0.1",
    "Pinball_0.25",
    "Pinball_0.75",
    "Pinball_0.9",
]

DATATSETS = {
    "train" : "",
    "test" : "_2.",
    "val" : ""
}
#
# Daset bound train to val avoiding overlap of 42 hours and Databounds test fitting to model evaluation of dfr
#
DATATSET_BOUNDS = {
        "train" : (pd.to_datetime("2010-07-08 12:00:00"), pd.to_datetime("2012-02-02")),
        "val" : (pd.to_datetime("2012-02-04 12:00:00"), pd.to_datetime("2012-06-28")),
        "test" : (pd.to_datetime("2012-07-08 12:00:00"), pd.to_datetime("2013-06-28"))
    }

FACTORS = [
    "results_factor5", "results_factor10" ,"results_ldiv2" ,"results_ldiv5" ,"results_load2" ,"results_load5" ,"results"

]

RESULTS_FOLDER = "data_res/results_opti"
FORECAST_FOLDER = "data_res/results_forecast"
MODEL_SAVE_FOLDER = "models/models_optimisation/with_SoE/"

SCALER = None
SCALER_FITTED = False

FEATURE_SCALER = None
FEATURE_SCALER_FITTED = False

#
# Helper Method to get the data from the data folder
#

# get the data by building id and convert it to a dataframe
def get_data(id,factor=None):
    #print("Get Data for Building: ", id)

    if factor != None:
        # Get the cost data and the SoE at the beginning of the day (Replacement for estimated SoE)
        ds_costs = pickle.load(open("data/"+ RESULTS_FOLDER + "/" + factor + "/ds_costs_daily_optimisation/" + str(id), "rb"))
        imbalance_costs = pickle.load(open("data/"+ RESULTS_FOLDER + "/" + factor +"/imbalance_costs_daily_optimisation/" + str(id) , "rb"))

        # Get the SoE at the beginning of the day (Replacement for estimated SoE)
        train_soe = pickle.load(open("data/"+ RESULTS_FOLDER + "/" + factor +"/uni_soe_train/" + str(id) , "rb"))
    else:
        print("No Factor Specified")


    soe = {}
    soe["train"] = train_soe[0][DATATSET_BOUNDS["train"][0]:DATATSET_BOUNDS["train"][1]]
    soe["val"] = train_soe[0][DATATSET_BOUNDS["val"][0]:DATATSET_BOUNDS["val"][1]]
    soe["test"] = pickle.load(open("data/"+ RESULTS_FOLDER + "/" + factor + "/uni_soe_test/" + str(id) , "rb"))[0][DATATSET_BOUNDS["test"][0]:DATATSET_BOUNDS["test"][1]]

    # Get the total costs for val train and test

    total_costs = {}
    total_costs["val"] = {}
    for key in ds_costs:
        total_costs[key] = {}
        for loss_function in ds_costs[key]:
            if key == "train":
                tmp_key = "val"
                total_costs_per_loss = ds_costs[key][loss_function] +IMBALANCE_FACTOR* imbalance_costs[key][loss_function]
                # VAL and Train
                total_costs[key][loss_function] = total_costs_per_loss[DATATSET_BOUNDS[key][0]:DATATSET_BOUNDS[key][1]]
                total_costs[tmp_key][loss_function] = total_costs_per_loss[DATATSET_BOUNDS[tmp_key][0]:DATATSET_BOUNDS[tmp_key][1]]
            else:
                total_costs_per_loss = ds_costs[key][loss_function] + IMBALANCE_FACTOR* imbalance_costs[key][loss_function]
                total_costs[key][loss_function] = total_costs_per_loss[DATATSET_BOUNDS[key][0]:DATATSET_BOUNDS[key][1]]

    # Get the forecast data
    forecasts = {}
    for dataset in DATATSETS:
        forecasts[dataset] = {}
        for method in FORECASTING_METHOD:
            forecasts[dataset][method] = {}
            for loss_function in LOSS_FUNCTION:
                if factor != None:
                    forecast = pd.read_csv(glob.glob("data/"+ FORECAST_FOLDER + "/" + str(factor) + "/"+ str(id)+ "/*" "/result_" + method + "_" + loss_function + DATATSETS[dataset]  + ".csv")[0], index_col=0, parse_dates=True)
                else:
                    print("No Factor Specified")
                
                forecasts[dataset][method][loss_function] = forecast[DATATSET_BOUNDS[dataset][0]:DATATSET_BOUNDS[dataset][1]]

    # get the ground truth data
    ground_truth = {}
    for dataset in DATATSETS:
        ground_truth[dataset] = {}
        if factor != None:
            ground_truth[dataset] = pd.read_csv(glob.glob("data/"+ FORECAST_FOLDER + "/" + str(factor) +"/"+ str(id)+ "/*/gt_forecast" + DATATSETS[dataset]  + ".csv")[0], index_col=0,parse_dates=True)[DATATSET_BOUNDS[dataset][0]:DATATSET_BOUNDS[dataset][1]]
        else:
            print("No Factor Specified")


    # get basic statistical information about the building

    min = ground_truth["train"].min().values[0]
    max = ground_truth["train"].max().values[0]
    mean = ground_truth["train"].mean().values[0]
    std = ground_truth["train"].std().values[0]


    # create a dataframe with the statistical information
    statistical_information = pd.DataFrame({"min": min, "max": max, "mean": mean, "std": std}, index=[0])

    
    # returning a dictionary with the data
    data = {}
    data["total_costs"] = total_costs
    data["soe"] = soe
    data["forecasts"] = forecasts
    data["ground_truth"] = ground_truth
    data["statistical_information"] = statistical_information

    return data

#
# Dataset Class for the OpNet
#


class OpNetDataset(Dataset):
    def __init__(self, building_range, dataset, factors):
        # A List with all the building data
        data_list = []
        for i in building_range:
            for factor in factors:
                data_list.append((get_data(i, factor=factor),i))

        first_building = True


        self.features = None
        self.value = None
        for building, building_id in data_list:
            #print("Building: ", building_id)
            for method in FORECASTING_METHOD:
                for loss_function in LOSS_FUNCTION:
                    
                    index = [i for i in range(0,len(building["total_costs"][dataset][loss_function].values))]
                    # get the forecasted values
                    forecast = building["forecasts"][dataset][method][loss_function]
                    # get the ground truth values
                    gt = building["ground_truth"][dataset]
                    # get the SoE at the beginning of the day
                    soe = building["soe"][dataset]

                    stats = building["statistical_information"]

                    # get the total costs
                    total_costs = pd.DataFrame(building["total_costs"][dataset][loss_function].values, columns=["total_costs"])


                    forecast["index"] = index
                    gt["index"] = index
                    soe["index"] = index
                    total_costs["index"] = index

                    # expand the soe dataframe with the statistical information
                    for x in building["statistical_information"].columns:    
                        soe[x] = np.full((len(index)),stats[x].values[0])
                    soe_stats = soe.set_index("index")

                    forecast = forecast.set_index("index")
                    gt = gt.set_index("index")
                    total_costs = total_costs.set_index("index")


                    if first_building:

                        self.features = pd.concat([forecast, gt, soe_stats], axis=1)
                        self.value = total_costs
                        first_building = False

                    else:
                        tmp_features = pd.concat([forecast, gt, soe_stats], axis=1)
                        tmp_value = total_costs
                        self.features = pd.concat([self.features, tmp_features], axis=0)
                        self.value = pd.concat([self.value, tmp_value], axis=0)

        # scale the output data 
        global SCALER
        global SCALER_FITTED
        if SCALER_FITTED == False:
            SCALER.fit(self.value)
            SCALER_FITTED = True

        # scale the features 

        

        global FEATURE_SCALER_FITTED
        global FEATURE_SCALER
        
        if FEATURE_SCALER_FITTED == False:
            FEATURE_SCALER.fit(self.features.values[:,84:])
            FEATURE_SCALER_FITTED = True
            #save the scaler in a seperate pickle file
            #pickle.dump(FEATURE_SCALER, open("models/feature_scaler/feature_scaler.pkl", "wb"))  


        
        
        fig = px.histogram(self.value.values, nbins=100)
        #plt.title(f"Histogram of the total costs after scaling | Dataset: {dataset}")
        wandb.log({f"Total costs before {dataset}": fig})

        self.value = SCALER.transform(self.value.values)
        self.features.iloc[:,84:] = FEATURE_SCALER.transform(self.features.values[:,84:])

        fig = px.histogram(self.value, nbins=100)
        #plt.title(f"Histogram of the total costs after scaling | Dataset: {dataset}")
        wandb.log({f"Total costs after {dataset}": fig})


        self.features = torch.tensor(self.features.values,dtype=torch.float32)
        self.value = torch.tensor(self.value,dtype=torch.float32)

                            
                    
    # return the length of the dataset
    def __len__(self):
        return len(self.features)
    # return the features and the value
    def __getitem__(self, idx):
        return self.features[idx], self.value[idx]
    # return the shape of the data
    def data_shape(self):
        return "Features: ", self.features.shape, "Value: ", self.value.shape

#
# Simple Neural Network using Pytorch used later as Op
#

class OptiEstimator(nn.Module):

    def __init__(self ,output_size):
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
        #split the last value from the input
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


#
# Main Method
#   
   
if __name__ == "__main__":



    config={
    "learning_rate": 0.001,
    "min_delta": 0.0001,
    "patience": 20,
    "architecture": "MLP",
    "train_buildings": [i for i in range(1,51)],
    "val_buildings": [i for i in range(51,101)],
    "test_buildings": [i for i in range(101,301)],
    "epochs": 100,
    "Datasets": DATATSETS,
    "Dataset_bounds": DATATSET_BOUNDS,
    "Imbalance_factor": IMBALANCE_FACTOR,
    "Forecasting_methods": FORECASTING_METHOD,
    "Loss_functions": LOSS_FUNCTION,
    "Results_folder": RESULTS_FOLDER,
    "Forecast_folder": FORECAST_FOLDER,
    "Model_save_folder": MODEL_SAVE_FOLDER,
    "Loss_function": "MAE",
    "Optimizer": "Adam",
    "Scaler": "QuantileTransformer",
    }


    #
    #  Removed convergence issues of the optimization problem
    #
    config["train_buildings"].remove(16) # load2 missing
    config["train_buildings"].remove(19) # load1div5 missing

    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="Op-Net",
    # track hyperparameters and run metadata
    config=config,
    )
    

    # build up a simple neural network using pytorch getting 85 input features and 1 output feature
    # the input features are the forecasted values for the next 42 hours
    # and the SoE at the beginning of the day
    # and the ground truth values for the last 42 hours
    # the output is the costs using the foreacasted values given the ground truth values and the SoE at the beginning of the day

    # Set the scaler in advance
    if wandb.config["Scaler"] == "QuantileTransformer": 
        SCALER = SklearnQuantileTransformer(output_distribution='uniform')

    # Set Scaler for the features
    FEATURE_SCALER = SklearnMinMaxScaler(feature_range=(0,1),clip=True)

    training_data = OpNetDataset(wandb.config["train_buildings"], "train", FACTORS)
    print("Training Data :" , training_data.data_shape())

    validation_data = OpNetDataset(wandb.config["val_buildings"], "val" , ["results"])
    print("Validation Data :" , validation_data.data_shape())

    test_data = OpNetDataset(wandb.config["test_buildings"], "test", ["results"])
    print("Test Data :" , test_data.data_shape())

    print("Start Training")
    # define a simple neural network using pytorch  
    network = OptiEstimator(1)
    # define the optimizer


    if wandb.config["Optimizer"] == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=wandb.config["learning_rate"])
    # define the loss function

    loss_function = None
    if wandb.config["Loss_function"] == "MAE":    
        loss_function = nn.L1Loss()
    
    # define the batch size
    batch_size = 256
    
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
    
    early_stopper = EarlyStopper(patience=config["patience"], min_delta=config["min_delta"])

    network.train()
    network.to("cuda")
    wandb.watch(network, log='all')

    tmp_model = None
    val_losses_epoch = []
    # define the training loop
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            losses = []
            # get the inputs
            x_batch, y_batch = data

            x_batch = x_batch.to("cuda")    
            y_batch = y_batch.to("cuda")    
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(x_batch)
            loss = loss_function(outputs, y_batch)


            ### REGULARIZATION ###

            # add l1 regularization
            loss += l1_regularizer(network, lambda_l1=0.00000001) 

            # add l2 regularization
            loss += l2_regularizer(network, lambda_l2=0.00000001)


            # backpropagate the loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


        # print statistics
        print("Epoch: ", epoch, " Loss: ", np.asarray(losses).mean())

        val_losses = []
        # define the validation loop
        for i, data in enumerate(val_loader, 0):
            val_losses = []
            # get the inputs
            x_batch, y_batch = data
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
            tmp_model = copy.deepcopy(network)
        
        if early_stopper.early_stop(epoch_val_loss): 
            print("Early Stopping, restoring the best weights")
            network = tmp_model            
            break
    network = tmp_model       
    
    # define the test
    test_loss_MSE = []
    test_loss_MAE = []

    test_loss_MAE_scaled = []
    test_loss_MSE_scaled = [] 
    for i, data in enumerate(test_loader, 0):
        # get the inputs

        x_batch, y_batch = data
        x_batch = x_batch.to("cuda")    
        y_batch = y_batch.to("cuda")    
        
        with torch.no_grad():
            outputs = network(x_batch)

            loss_function_MSE = nn.MSELoss()
            loss_function_MAE = nn.L1Loss()

            loss_MSE_scaled = loss_function_MSE(torch.tensor(outputs), torch.tensor(y_batch))
            loss_MAE_scaled = loss_function_MAE(torch.tensor(outputs), torch.tensor(y_batch))
            test_loss_MSE_scaled.append(loss_MSE_scaled.item())
            test_loss_MAE_scaled.append(loss_MAE_scaled.item())

            outputs = SCALER.inverse_transform(outputs.cpu().numpy())
            y_batch = SCALER.inverse_transform(y_batch.cpu().numpy())

            loss_MSE = loss_function_MSE(torch.tensor(outputs), torch.tensor(y_batch))
            loss_MAE = loss_function_MAE(torch.tensor(outputs), torch.tensor(y_batch))
            test_loss_MSE.append(loss_MSE.item())
            test_loss_MAE.append(loss_MAE.item())


    print("Test Evaluation: ")
    print("Test Loss MSE: ", np.asarray(test_loss_MSE).mean())
    print("Test Loss MAE: ", np.asarray(test_loss_MAE).mean())
    print("Test Loss MSE Scaled: ", np.asarray(test_loss_MSE_scaled).mean())
    print("Test Loss MAE Scaled: ", np.asarray(test_loss_MAE_scaled).mean())


    # log the loss and accuracy values at the end 
    wandb.log({"test_loss_MSE": np.asarray(test_loss_MSE).mean(), "test_loss_MAE": np.asarray(test_loss_MAE).mean(), "test_loss_MSE_scaled": np.asarray(test_loss_MSE_scaled).mean(), "test_loss_MAE_scaled": np.asarray(test_loss_MAE_scaled).mean()})  
    # save the model with time stamp in a models folder which it creates if it does not exist
    name = "model_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".pt"
    torch.save(network.state_dict(), MODEL_SAVE_FOLDER + name)
    wandb.log({"network_saved_as": name})
    

    # save the model in the wandb cloud storage
    torch.save(network.state_dict(), wandb.run.dir + "/model.pt")
    # mark the run as finished
    wandb.finish()

    

    


    

    





          
