# Decision-Focused Retraining of Forecast Models for Optimization Problems in Smart Energy Systems

This repository contains the Python implementation, the surrogate networks for the paper:
> Maximilian Beichter, Dorina Werling, Benedikt Heidrich, Kaleb Phipps, Oliver Neumann, Nils Friederich, Ralf Mikut und Veit Hagenmeyer. 2024.
> Decision-Focused Retraining of Forecast Models for Optimization Problems in Smart Energy Systems. In The 15th
> ACM International Conference on Future Energy Systems (e-Energy ’24). ACM, pp. tbd. https://doi.org/10.1145/3632775.3661952

## Repository Structure

- 'src': This folder contains the code used for retraining and the training of the optimisation problem.
    - Subfolder 'modules' contains some used torch implementations
    - Subfolder 'optimisation' contains the optimisation problem code
    - File 'main_optinet_training.py' creates Surrogate Neural Networks
    - File 'main_retraining_MSE.py' contains the main file for the retraining process with MSE Loss
    - File 'main_retraining_Pinball.py' contains the main file for the retraining process with Pinball Loss
    - File 'main_optimisation.py' and 'main_optimisation_gt.py' contain the code to launch the optimisation problem.
- 'run_configs': This folder contains shell scripts to run the code.
- 'models': This folder contains the surrogate networks and the feature scaler. 
- 'data': This folder contains the data used for training tests and validation.
- 'results': This folder will be created through the software and contains the forecasting results.
- 'results_optimisation': This folder will be created through the software and contains the optimisation results. 

## Installation


### 1. Setup Python Environment
- 1. Install virtualenv (if not already installed)
```
pip install virtualenv
```
- 2. Create a Virtual Environment
```
virtualenv myenv
```
- 3. Activate the Virtual Environment
```
source myenv/bin/activate
```
- 4. Install Packages from requirements.txt
```
pip install -r requirements.txt
```

Further, make sure that Ipopt is properly installed. For more details, look here:
[IPOPT](https://github.com/coin-or/Ipoptr)

## Execution

Disclaimer. The actual code is not runnable, as the data is missing due to a missing data statement of the [Ausgrid Dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) we are using.

You can generate similar data following the 'generate_data_setup.iypnb' in addition to this you need to build up own forecasts, and run the optimisation problem by using the file "src/main_optimisation_dataset_creation.py" 

## Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, the Helmholtz Association under the Program “Energy System Design”, and the German Research Foundation (DFG) as part of the Research TrainingGroup 2153 “Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation”. This work is supported by the Helmholtz Association Initiative and Networking Fund on the HAICORE@KIT partition.

## License

This code is licensed under the [MIT License](LICENSE).