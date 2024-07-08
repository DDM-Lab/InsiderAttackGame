from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

import os
import sys

import math
import matplotlib .pyplot as plt
import csv
import pandas as pd
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(script_dir,"Generated_Models_Data")



def GenerateConfusionMatrix(actual_decisions, predicted_decisions, label, tracing, model, path):
  
    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        #file_path = os.path.join(script_dir, "Plots_Tracing_ConfusionMatrixes_CD\\Individual")
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"CM_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        #file_path = os.path.join(script_dir, "Plots_NoTracing_ConfusionMatrixes_CD\\Individual")
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"CM_NoTracing_{label}_{model}_{current_datetime}.png"
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    # Construct the confusion matrix
    cm = confusion_matrix(actual_decisions, predicted_decisions, labels=["Attack", "Withdrew"], normalize='all')


    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Extract values from confusion matrix
    true_positives = cm[0, 0]  # Attack correctly predicted as Attack
    true_negatives = cm[1, 1]  # Withdrew correctly predicted as Withdrew
    total_instances = np.sum(cm)  # Sum of all instances


    # Calculate accuracy
    accuracy = round((true_positives + true_negatives) / total_instances,2)

    print(f'Accuracy: {accuracy}')



    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Attack", "Withdrew"])
    disp.plot()

    ax = disp.ax_

    #Adding labels to the four positions
    for i, text in enumerate(ax.texts):
    # Modify the text as needed
        x = text.get_position()[0]
        y= text.get_position()[1]
        text_str = text.get_text()

        # Convert the string to a float
        float_value = float(text_str)

        # Truncate the float to 2 decimal places
        truncated_float_value = int(float_value * 100) / 100 #float("{:.2f}".format(float_value))
        
        new_text = f"{truncated_float_value}"
        text.set_text(new_text)


    if "MachineCovered" in label or "MachineNotCovered" in label:
        plt.xlabel("Model Predictions")  # Label for x-axis
        plt.ylabel("Human Choices")  # Label for y-axis
        plt.title(f"Target Covered Ground Truth = {label}\n Accuracy = {accuracy}, {model}\nTracing = {tracing}")  # Title of the plot
    elif "Human-Model" in label:
        plt.xlabel("Model Predictions")  # Label for x-axis
        plt.ylabel("Human Choices")  # Label for y-axis
        plt.title(f"Syncronization {label}\n Accuracy = {accuracy}, {model}\nTracing = {tracing}")   # Title of the plot
    else:
        if "Human" in label:
            plt.xlabel("Human Choices")  # Label for x-axis
            plt.title(f"{label} \n Accuracy = {accuracy}")  # Title of the plot
        else:
            plt.xlabel("Model Predictions")  # Label for x-axis
            plt.title(f"{label} \n Accuracy = {accuracy}, {model}\nTracing = {tracing}")  # Title of the plot
        plt.ylabel("Ground Truth")  # Label for y-axis
        
    plt.tight_layout()

    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

    return plot_path
    


def GenerateLearningCurves(predicted_decisions, label, tracing, model, path):
  
    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        #file_path = os.path.join(script_dir, "Plots_Tracing_LearningCurves\\Individual")
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"LC_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        #file_path = os.path.join(script_dir, "Plots_NoTracing_LearningCurves\\Individual")
        file_path = os.path.join(script_dir, f"Plots_{path}\\Individual")
        plotName = f"LC_NoTracing_{label}_{model}_{current_datetime}.png"
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    plt.figure()  # Create a new figure

    
    # Convert 'TrialNumber' column to numeric, coercing errors to NaN

    predicted_decisions.loc[:,('TrialNumber')] = pd.to_numeric(predicted_decisions['TrialNumber'], errors='coerce')

    # Add 25 to TrialNumber where TrainingBlock is 4
    predicted_decisions.loc[predicted_decisions['TrainingBlock'] == 4, 'TrialNumber'] += 25
   
    accuracy = predicted_decisions.groupby(["TrialNumber", "TrainingBlock"]).agg(
    avg_modelSyncUser=('modelSyncUser', 'mean'),
    ).reset_index()

    predicted_decisions.loc[predicted_decisions['TrainingBlock'] == 4, 'TrialNumber'] -= 25

    plt.plot(accuracy['TrialNumber'], accuracy['avg_modelSyncUser'])
        
    
    # Set y-axis range between 0 and 1
    plt.ylim(0, 1)

    # Add vertical dashed line at trial 25
    plt.axvline(x=25, color='grey', linestyle='--')

    # Add horizontal dashed line at y=0.5 with grey color
    plt.axhline(y=0.5, color='grey', linestyle='--')
    
    plt.xlabel("Trials")  # Label for x-axis
    plt.ylabel("Average of SyncRate")  # Label for y-axis

    if "MachineCovered" in label or "MachineNotCovered" in label:
        plt.title(f"Learning Curve \n Target Covered Ground Truth = {label}\n {model}\nTracing = {tracing}")  # Title of the plot
    elif "Human-Model" in label:
        plt.title(f"Learning Curve\n Syncronization {label}\n {model}\nTracing = {tracing}")   # Title of the plot
    else:
        if "Human" in label:
            plt.title(f"Learning Curve {label}\n")  # Title of the plot
        else:
            plt.title(f"Learning Curve {label}\n {model}\nTracing = {tracing}")  # Title of the plot
    
    plt.tight_layout()

    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

    return plot_path

def CombinePlots(dataPlots, rows, cols, title, tracing, model, label, path):

    # Generate current date and time
    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')

    if tracing:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Grouped")
        if "ConfusionMatrixes" in path:
            plotName = f"CM_Tracing_{label}_{model}_{current_datetime}.png"
        else:
            plotName = f"LC_Tracing_{label}_{model}_{current_datetime}.png"
    else:
        file_path = os.path.join(script_dir, f"Plots_{path}\\Grouped")
        if "ConfusionMatrixes" in path:
            plotName = f"CM_NoTracing_{label}_{model}_{current_datetime}.png"
        else:
            plotName = f"LC_NoTracing_{label}_{model}_{current_datetime}.png"
        
    os.makedirs(file_path, exist_ok=True)

    plot_path = os.path.join(file_path,plotName)

    # Combine plots
    if rows == 1:

        #fig, axes = plt.subplots(rows, cols, figsize=(14, 4))
        fig, axes = plt.subplots(rows, cols, figsize=(6, 3))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(8, 9))

    for i, plot in enumerate(dataPlots):
        row = i // cols
        col = i % cols
        if rows == 1:
            img = plt.imread(plot)
            axes[col].imshow(img)
            axes[col].axis('off')  # Hide axes for individual plots
        else:
            img = plt.imread(plot)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')  # Hide axes for individual plots
        

   # Set title for the entire plot
    plt.suptitle(f'{title}')
    plt.tight_layout()  # Adjust layout
    plt.savefig(f'{plot_path}')  # or .jpg, .pdf, etc.
    plt.show()
    plt.close()

if __name__ == "__main__":

    dataFile = ["Tracing_Results\IAG_Tracing_Data_Personalized.csv"]

    plotsStep1_CM = []
    plotsStep2_CM = []
    

    plotsStep1_LC = []
    plotsStep2_LC = []
    

    for f in dataFile:

        file_path = os.path.join(DATA_FILE,f)

        predictedData = pd.read_csv(file_path)
        Covered_data = []
        NotCovered_data = []
        PersonalizedData = []

        if "No_Tracing" in f:
            tracing = False
            model_type="Training_50_50 Default Decay and Noise"
        else:
            tracing = True
            if "Personalized" in f:
                model_type="Training_50_50 Personalized Decay"
            else:
                model_type="Training_50_50 Default Decay"

        
        # Filter data where 'Target Covered Ground Truth' is 'Covered'
        Covered_data = predictedData[(predictedData['TargetCoveredGroundTruth'] == 'Yes') & 
                                    ((predictedData['TrainingBlock'] == 3) |
                                        (predictedData['TrainingBlock'] == 4))]
        
        # Filter data where 'Target Covered Ground Truth' is 'NotCovered'
        NotCovered_data = predictedData[(predictedData['TargetCoveredGroundTruth'] == 'No') & 
                                    ((predictedData['TrainingBlock'] == 3) |
                                        (predictedData['TrainingBlock'] == 4))]
        

        PersonalizedData = predictedData[((predictedData['TrainingBlock'] == 3) |
                                        (predictedData['TrainingBlock'] == 4))]

        
        


        #################### Plot Model-Human Data  all training phases together

        human_predicted =  np.array(PersonalizedData['UserChoice']).astype(str)
        model_predicted  = np.array(PersonalizedData['ModelChoice']).astype(str)
        plot_CM = GenerateConfusionMatrix(human_predicted , model_predicted ,'Human-Model',tracing, model_type, "ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(PersonalizedData ,'Human-Model',tracing, model_type, "LearningCurves")

        plotsStep1_CM.append(plot_CM)
        plotsStep1_LC.append(plot_LC)

        
        #################### All training phases together


        # Filter data where 'Target Covered Ground Truth' is 'Covered' 

        MachineCoveredHuman =  np.array(Covered_data['UserChoice']).astype(str)
        MachineCoveredModel = np.array(Covered_data['ModelChoice']).astype(str)

        plot_CM = GenerateConfusionMatrix(MachineCoveredHuman, MachineCoveredModel, 'MachineCovered', tracing, model_type, "ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(Covered_data, 'MachineCovered', tracing, model_type, "LearningCurves")

        Covered_data['TrialNumber']

        plotsStep2_CM.append(plot_CM)
        plotsStep2_LC.append(plot_LC)
        
        # Filter data where 'Target Covered Ground Truth' is 'NotCovered'

        MachineNOTCoveredHuman =  np.array(NotCovered_data['UserChoice']).astype(str)
        MachineNOTCoveredModel = np.array(NotCovered_data['ModelChoice']).astype(str)

        plot_CM = GenerateConfusionMatrix(MachineNOTCoveredHuman, MachineNOTCoveredModel, 'MachineNotCovered', tracing, model_type, "ConfusionMatrixes")
        plot_LC = GenerateLearningCurves(NotCovered_data, 'MachineNotCovered', tracing, model_type, "LearningCurves")
        
        NotCovered_data['TrialNumber']

        plotsStep2_CM.append(plot_CM)
        plotsStep2_LC.append(plot_LC)

        
        #############################################################################################

        

        
        #############################################################################################
    ## When we are comparing more 3 model results       
    """ CombinePlots(plotsStep1_CM, 1, 3, "Model-Human Syncronization", tracing, model_type, 'Human-Model', "ConfusionMatrixes")
    CombinePlots(plotsStep1_LC, 1, 3, "Model-Human Syncronization", tracing, model_type, 'Human-Model', "LearningCurves")
    CombinePlots(plotsStep2_CM, 3, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "ConfusionMatrixes")
    CombinePlots(plotsStep2_LC, 3, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "LearningCurves")
     """

    CombinePlots(plotsStep2_CM, 1, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "ConfusionMatrixes")
    CombinePlots(plotsStep2_LC, 1, 2, "Model-Human Syncronization split by GroundTruth", tracing, model_type, 'Split_GroundTruth', "LearningCurves")

        
    sys.exit()

  