from optimisation.ClassOpti import Opti
import argparse
import pandas as pd
from glob import glob
import os
import pickle
import numpy as np


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)


def pickle_opimisation(optimisation, path):
    """

    Args:
        optimisation: optimisation instance
        path: path to safe the file contains res_path and the given forecast_path

    Returns:n nothing

    """
    # open a file, where you ant to store the data
    file = open(path, 'wb')

    # dump information to that file
    pickle.dump(optimisation, file)

    # close the file
    file.close()


if __name__ == '__main__':
    #
    # Argument Parser
    #

    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('-id', type=str, help='building_id', default="1")

    parser.add_argument('-factor',type=str, help='factor',default="")

    args = parser.parse_args()
    
    if args.factor != "":
        args.factor = "_" + args.factor

    #
    # Create result folder if not existent
    #
    res_path = f"data/data_res/results_opti/results{args.factor}/results_optimisation/{args.id}"
    make_dir(f"data/data_res/results_opti/results{args.factor}/results_optimisation/{args.id}")
    benchi_path = f"data/data_res/results_opti/results{args.factor}/results_benchmark/{args.id}"
    make_dir(f"data/data_res/results_opti/results{args.factor}/results_benchmark/{args.id}")

    #
    # read in the specific file
    #

    # UPDATE YOUR FORECAST PATH HERE 
    forecast_paths = glob(f"data/data_res/results_forecast/results{args.factor}/{args.id}/*/result_FC_with_Cov*")


    # should work
    actual_path_train = glob(f"data/data_res/results_forecast/results{args.factor}/{args.id}/*/gt_daily_daily.csv")
    actual_path_test = glob(f"data/data_res/results_forecast/results{args.factor}/{args.id}/*/gt_daily_daily_2..csv")

    # Setting fixed SoE
    soe_test = [pd.DataFrame(np.random.uniform(0, 13.5, len(pd.date_range(pd.Timestamp(year=2012, month=7, day=8),
                                                 pd.Timestamp(year=2013, month=6, day=28)))),
                             index=pd.date_range(pd.Timestamp(year=2012, month=7, day=8),
                                                 pd.Timestamp(year=2013, month=6, day=28)))]
    soe_train = [pd.DataFrame(np.random.uniform(0, 13.5, len(pd.date_range(pd.Timestamp(year=2010, month=7, day=8),
                                                  pd.Timestamp(year=2012, month=6, day=28)))),
                              index=pd.date_range(pd.Timestamp(year=2010, month=7, day=8),
                                                  pd.Timestamp(year=2012, month=6, day=28)))]

    # Initialise dictionaries for ds/imbalance costs for fixed soe optimisation and benchmark
    ds_costs = {'train': pd.DataFrame(), 'test': pd.DataFrame()}
    imbalance_costs = {'train': pd.DataFrame(), 'test': pd.DataFrame()}
    ds_costs_benchi = {'train': pd.DataFrame(), 'test': pd.DataFrame()}
    imbalance_costs_benchi = {'train': pd.DataFrame(), 'test': pd.DataFrame()}

    print(forecast_paths)
    # Optimisation
    for forecast_path in forecast_paths:

        opti = Opti(1)
        benchi = Opti(1)

        # Set paths/strings for saving
        path = os.path.join(res_path, forecast_path.split(os.sep)[-1][:-4])
        path_benchi = os.path.join(benchi_path, forecast_path.split(os.sep)[-1][:-4])
        if '_Pinball' in forecast_path:
            str_tempi = forecast_path.split(os.sep)[-1][:-4].split('_')
            str_loss = str_tempi[4] + "_" + str_tempi[5]
        else:
            str_loss = forecast_path.split(os.sep)[-1][:-4].split('_')[4]

        if "_2" in forecast_path.split(os.sep)[-1]:
            start_date = pd.to_datetime('2012-07-08')
            end_date = pd.to_datetime('2013-06-28')
            path_soe = glob(f"data/data_res/results_forecast/results{args.factor}/{args.id}/*/result_FC_with_Cov_MSE_2..csv")

            opti.run_deterministic_fixed_soe(soe_test, [forecast_path], actual_path_test, start_date, end_date)
            benchi.run_deterministic_different_forecasts([benchi.e_max["house0"] / 2], path_soe, [forecast_path], actual_path_test, start_date,
                                     end_date)
            print("test runned for ", str_loss)
            # Set string for saving
            str_dataset = 'test'


        else:
            start_date = pd.to_datetime('2010-07-08')
            end_date = pd.to_datetime('2012-06-28')

            # PATH TO YOUR MSE FORECAST HERE
            path_soe = glob(f"data/data_res/results_forecast/results{args.factor}/{args.id}/*/result_FC_with_Cov_MSE.csv")

            opti.run_deterministic_fixed_soe(soe_train, [forecast_path], actual_path_train, start_date, end_date)
            benchi.run_deterministic_different_forecasts([benchi.e_max["house0"] / 2], path_soe, [forecast_path], actual_path_train, start_date,
                                     end_date)
            print("train runned for ", str_loss)
            # Set string for saving
            str_dataset = 'train'

        # Saving instances
        pickle_opimisation(opti, path)
        pickle_opimisation(benchi, path_benchi)

        # Calculate costs
        opti.calculate_ds_costs_daily('house0')
        opti.calculate_imbalance_costs_daily('house0')
        benchi.calculate_ds_costs_daily('house0')
        benchi.calculate_imbalance_costs_daily('house0')

        # Saving costs in dict

        ds_costs[str_dataset] = pd.concat(
            [ds_costs[str_dataset], opti.ds_costs_daily.rename(str_loss)], axis=1)
        imbalance_costs[str_dataset] = pd.concat(
            [imbalance_costs[str_dataset], opti.imbalance_costs_daily.rename(str_loss)], axis=1)
        ds_costs_benchi[str_dataset] = pd.concat(
            [ds_costs_benchi[str_dataset], benchi.ds_costs_daily.rename(str_loss)],
            axis=1)
        imbalance_costs_benchi[str_dataset] = pd.concat(
            [imbalance_costs_benchi[str_dataset], benchi.imbalance_costs_daily.rename(str_loss)], axis=1)

    # Saving cost dicts

    import os
    make_dir(f"data/data_res/results_opti/results{args.factor}/ds_costs_daily_optimisation")
    make_dir(f"data/data_res/results_opti/results{args.factor}/imbalance_costs_daily_optimisation")
    make_dir(f"data/data_res/results_opti/results{args.factor}/ds_costs_daily_benchmark")
    make_dir(f"data/data_res/results_opti/results{args.factor}/imbalance_costs_daily_benchmark")


    path = f"data/data_res/results_opti/results{args.factor}/ds_costs_daily_optimisation/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(ds_costs, file)
    path = f"data/data_res/results_opti/results{args.factor}/imbalance_costs_daily_optimisation/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(imbalance_costs, file)
    path = f"data/data_res/results_opti/results{args.factor}/ds_costs_daily_benchmark/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(ds_costs_benchi, file)
    path = f"data/data_res/results_opti/results{args.factor}/imbalance_costs_daily_benchmark/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(imbalance_costs_benchi, file)

    # Saving uniform disti SoE

    make_dir(f"data/data_res/results_opti/results{args.factor}/uni_soe_test")
    make_dir(f"data/data_res/results_opti/results{args.factor}/uni_soe_train")

    path = f"data/data_res/results_opti/results{args.factor}/uni_soe_test/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(soe_test, file)
    path = f"data/data_res/results_opti/results{args.factor}/uni_soe_train/{args.id}.pkl"
    with open(path, 'wb') as file:
        pickle.dump(soe_train, file)




