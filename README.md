# InsiderAttackGame

Aggarwal, P., Cranford, E.A., Tambe, M., Lebiere, C., Gonzalez, C. (2023). Deceptive Signaling: Understanding Human Behavior Against Signaling Algorithms. In: Bao, T., Tambe, M., Wang, C. (eds) Cyber Deception. Advances in Information Security, vol 89. Springer, Cham. https://doi-org.cmu.idm.oclc.org/10.1007/978-3-031-16613-6_5

The Insider Attack Game is a simulated real-world scenario of a network composed of six nodes (computers) with resource constranined alocation where a defender can only protect two nodes. The game evolves over four phases, each with 25 decisions. Each decision comprises two stages in which participants play the role of attackers. In the first stage of the decision, participants must choose a node to attack while defenders protect nodes (computers) by randomly allocating limited resources. In the second stage of the decision, the defender decides probabilistically to either warn the attacker that the node is defended (which could be a lie) or not provide such a warning, and then the attacker (participant) chooses to continue the attack or not.

## Data files

The human experiment dataset are available on OSF: https://osf.io/c7ntu/?view_only=0e6261b5e818440495d9917044611758

For this particular implementation we are using:

    - 2022-MURIBookChapter-FullData-IAG.csv
    - AllConditions_humanData_dictionary.txt (dictionary with variables information)

`max_decays_IAG.csv` was generated after running Model-Fitting for each model containing the best decay for each participant.



## IBL model

- `insiderAttackGame.py` contains the implementation of the IBL model for model-tracing. 
    - To run this code, you will need the two files located inside the folder `Data` at the root of the project. One of the files can be found in the OSF link, and the other is automatically generated after running `runFitting.py` script.

- `confusionMatrix.py` contains functions to generate confusion matrixes for both models in different settings. 
    - The data used in this code is automatically generated by the `insiderAttackGame.py` script by creating a folder named `Generated_Models_Data` with the necessary files.

- `runFitting.py` has the development of the Model-Fitting implementation according to the IBL model in the `insiderAttackGame.py`.
    - In this script, each participant's data will be run by 291 IBL models, and each model will have a `decay value` in the range of `[0.1 - 3]` with increments of `0.01`.
    - To run this code, you will need the file `2022-MURIBookChapter-FullData-IAG.csv` located in the `Data` folder. 
    - This script will generate automatically the results of all the models per participant and record it as `Fitting_Tracing_Data.csv` in the folders `Generated_Models_Data\Fitting_Tracing_Results` on the root of the project. 
    
- `IAG.Rmd` is an `R code` script used to assess the best fitting decay for each participant based on the 291 Models generated. 
    - The best fitting is assessed based on the highest `Synchronization Rates (SyncRate)` of each model for each participant. If multiple models have the same SyncRate, the model with the highest decay is selected for that participant. 
        - `SyncRate` examines the synchronization between the model prediction and each human choices. We determined whether the model prediction was the same as the actual human action for each decision. If it was the same, the synchronization value for that decision was `1`; otherwise, it was `0`. 
        - We calculated the `SyncRate` average per participant for each IBL model (with a different decay value) for the last 50 decisions (the model only predicts the last 50 decisions).
    - To run this script, you must first run the `runFitting.py` script and use the generated `Fitting_Tracing_Data.csv` file.
    - This script will automatically generate the `max_decays_IAG.csv` in the same folder where the script is located (same file as the one in the `Data` folder).

    **NOTE:** The R script was run in `Rstudio 2024.04.2+764` and needs an independent project. A copy of the file `Fitting_Tracing_Data.csv` needs to be added inside a folder named `Data\IAG_Game` in the root of the R project.


## Installation

Ensure Python version 3.8 or later is installed.

Run `python3 -m venv venv` to create a virtual environment.

Run `source env/bin/activate` to activate the virtual environment.

Run `pip install -r requirements.txt` to install the required Python packages into the virtual environment.

**NOTE:** If you do not want to create a virtual enviroment, just run `pip install -r requirements.txt`.



## More information

For more information or details, please contact [ddmlab@andrew.cmu.edu](mailto:ddmlab@andrew.cmu.edu).