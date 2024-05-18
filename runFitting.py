from pyibl import Agent
from random import random
from datetime import datetime

from multiprocessing import Process

import csv
import pandas as pd
import numpy as np
import os
import sys
import json
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir,"Data")


def runModel(experimentFile, tracing, decay, best_decayFile, noise):
    # Initialize a dictionary to store participant data for each decay
    decay_data = []

    file_path = os.path.join(DATA_FILE, experimentFile)
    ## read the experiment file
    expData = pd.read_csv(file_path)

    if len(best_decayFile) > 0:
        file_path = os.path.join(DATA_FILE, best_decayFile)
        best_decay = pd.read_csv(file_path)
    else:
        best_decay = []


    expData["Block"] = pd.to_numeric(expData["Block"], errors="coerce")
    expData["Trial"] = pd.to_numeric(expData["Trial"], errors="coerce")


    # Replace values in the "email_type" column
    expData['Warning'] = expData['Warning'].replace({0: 'Absent', 1: 'Present'})
    expData['Covered'] = expData['Covered'].replace({0: 'No', 1: 'Yes'})
    expData['Action'] = expData['Action'].replace({0: 'Withdrew', 1: 'Attack'})


    # Concatenating "MturkID" and "Condition" columns
    expData['MturkID_Condition'] = expData['MturkID'].astype(str) + '_' + expData['Condition']


    ## order the data by "Mturk_id", "phase", "trial"
    column_order = ["MturkID_Condition", "Block", "Trial"]

    

    # Sort the DataFrame based on the specified columns
    expData = expData.sort_values(by=column_order)

    writer = []
    for p in range(0,len(expData)):

        userID = expData['MturkID'][p]
        userBlock = expData['Block'][p]
        userTrial = expData['Trial'][p]

        userChoosedMachineNumber = expData['TargetNum'][p]
        userChoosedMachineLocation = expData['Best_Location'][p]

        userPayment= expData['Best_Payment'][p]
        userPenalty= expData['Best_Penalty'][p]
        userMprob= expData['Best_Mprob'][p]

        userReceivedSignal = expData['Warning'][p]
        userAction = expData['Action'][p]
        userOutcome = expData['Outcome'][p]

        userTargetCover = expData['Covered'][p]
        userCondition = expData['Condition'][p]

        if userBlock == 1 or userBlock == 2:
            if userBlock == 1 and userTrial == 1:
               ## Model with fitting parameters

                if tracing:
                    IBL_Agent = Agent(["TargetNum","TargetLocation","Payment","Penalty","Mprob", "Signal", "Decision"], name=f"{userCondition}_{userID}",
                                default_utility=0.0, noise = False,
                                decay = decay, mismatch_penalty=1.0, 
                                default_utility_populates = True)
                    noise = '-'
                else:
                    IBL_Agent = Agent(["TargetNum","TargetLocation","Payment","Penalty","Mprob", "Signal", "Decision"], name=f"{userCondition}_{userID}",
                            default_utility=0.0, noise = noise,
                            decay = decay, mismatch_penalty=1.0, 
                            default_utility_populates = True)
                
                IBL_Agent.reset()

                IBL_Agent.similarity(["Payment","Penalty"], lambda x, y: 1 - abs(x - y) / 10)
                IBL_Agent.similarity(["Mprob"], lambda x, y: 1 - abs(x - y))
            
            ## Prepopulate agent memory with all decisions from Block 1 and Blcok 2
            IBL_Agent.populate([(userChoosedMachineNumber,userChoosedMachineLocation,
                                 userPayment, (userPenalty * -1), userMprob,
                                 userReceivedSignal, userAction)], userOutcome)
            
            writer = WriteDataToDictionary(IBL_Agent.name, userCondition, userID, userBlock, userTrial,
                                           userChoosedMachineNumber,userChoosedMachineLocation, userReceivedSignal,
                                           userTargetCover,userAction, userAction, userOutcome, writer, tracing, decay, noise, '-', '-', '-', '-')
                
        if userBlock == 3 or userBlock == 4:
           
            modelChoice, d_choice = IBL_Agent.choose([(userChoosedMachineNumber,userChoosedMachineLocation,
                                 userPayment, (userPenalty * -1), userMprob,
                                 userReceivedSignal, 'Attack'),
                                 (userChoosedMachineNumber,userChoosedMachineLocation,
                                 userPayment, (userPenalty * -1), userMprob,
                                 userReceivedSignal, 'Withdrew')], True)

            for details in d_choice:
                op = details['choice']
                if op[6] == 'Attack':
                    bl_Attack = round(details['blended_value'], 3)
                    pRetrieval_Attack = json.dumps(convert_int64_to_int(details['retrieval_probabilities']))
                else:
                    bl_Withdrew = round(details['blended_value'], 3)
                    pRetrieval_Withdrew = json.dumps(convert_int64_to_int(details['retrieval_probabilities']))

            if tracing:
                payoff = userOutcome
                if modelChoice[6] == userAction:
                    IBL_Agent.respond(payoff)
                else:
                    IBL_Agent.respond(payoff, (userChoosedMachineNumber,userChoosedMachineLocation,
                                    userPayment, (userPenalty * -1), userMprob,
                                    userReceivedSignal, userAction))
            else:
                if modelChoice[6] == "Withdrew":
                    payoff = 0
                else:
                    payoff = (userPenalty * -1) if userTargetCover == 'Yes' else userPayment
                
                IBL_Agent.respond(payoff)

            writer = WriteDataToDictionary(IBL_Agent.name, userCondition, userID, userBlock, userTrial,
                                           userChoosedMachineNumber,userChoosedMachineLocation, userReceivedSignal,
                                           userTargetCover,modelChoice[6], userAction, payoff, writer, tracing, decay, noise, bl_Attack, bl_Withdrew, 
                                           pRetrieval_Attack, pRetrieval_Withdrew)


    # Convert the array to a DataFrame
    df_to_append = pd.DataFrame(writer)  

    # Specify the CSV file name
    if tracing and len(best_decay)== 0:
        csv_file_name = f'Fitting_Tracing_Data.csv'
    elif tracing and len(best_decay) > 0:
        csv_file_name = f'Tracing_Data_Personalized.csv'
    else:
        csv_file_name = 'IBL_Data_No_Tracing.csv'

    # Specify the desired header
    desired_header = ['Agent','Condition','Mturk_id','TrainingBlock','TrialNumber','MachineNumber','MachineLocation',
            'UserReceivedSignal','TargetCoveredGroundTruth','ModelChoice','UserChoice',
            'modelSyncUser','OutcomeReinforced','Tracing','decay_value','noise_value', 'Blended_Value_Attack', 'Blended_Value_Withdrew',
            'Retrieval_Probabilities_Attack', 'Retrieval_Probabilities_Withdrew']

    
    # Check if the CSV file already exists
    try:
        # Read the existing CSV file
        if tracing:
            #file_path = os.path.join(script_dir, "Tracing_Results")
            file_path = os.path.join(script_dir, "Generated_Models_Data", "Fitting_Tracing_Results")
        else:
            file_path = os.path.join(script_dir, "Generated_Models_Data", "IBL_Results")
        os.makedirs(file_path, exist_ok=True)
        existing_df = pd.read_csv(os.path.join(file_path, csv_file_name))

        # Append the new data to the existing DataFrame
        updated_df = df_to_append

    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        updated_df = df_to_append

    # Check if the header needs to be added
    if not os.path.isfile(os.path.join(file_path, csv_file_name)):
        # If the file doesn't have a header, add it
        updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='w', index=False, header=desired_header)
    else:
        # If the file already has a header, append without writing the header again
        updated_df.to_csv(os.path.join(file_path, csv_file_name), mode='a', index=False, header=False)

# Function to convert int64 to int in a nested structure
def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(elem) for elem in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def WriteDataToDictionary(AgentName,condition, userID, userBlock, userTrial, userChoosedMachineNumber,userChoosedMachineLocation, UserReceivedSignal,
                          TargetCovered, choiceModelMade, userDecision, outcomeReinforced, writer, tracing, decay, noise, 
                          bl_Attack, bl_Withdrew, pRetrieval_Attack, pRetrieval_Withdrew):

  writer.append({'Agent': AgentName,
                'Condition': condition,
                'Mturk_id': userID,
                'TrainingBlock':userBlock,
                'TrialNumber': userTrial,
                'MachineNumber':userChoosedMachineNumber,
                'MachineLocation':userChoosedMachineLocation,
                'UserReceivedSignal': UserReceivedSignal, 
                'TargetCoveredGroundTruth': TargetCovered,
                'ModelChoice': choiceModelMade, 
                'UserChoice': userDecision,
                'modelSyncUser': 1 if choiceModelMade == userDecision else 0,
                'OutcomeReinforced': outcomeReinforced,
                'Tracing':tracing,
                'decay_value':decay,
                'noise_value':noise,
                'blended_value_Attack': bl_Attack, 
                'blended_value_Withdrew': bl_Withdrew,
                'probRetrieval_Attack':pRetrieval_Attack, 
                'probRetrieval_Withdrew':pRetrieval_Withdrew})
  
  return writer


if __name__ == "__main__":


  experimentFile = "2022-MURIBookChapter-FullData-IAG.csv"
  decay_data = []

  start = 0.10
  end = 3.00
  step = 0.01

  d_values = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]
 

  try:
    for decay_value in d_values:
        runModel(experimentFile, tracing=True, decay=decay_value, best_decayFile=[], noise=0.25)

  except Exception as e:
    print(f"Error retrieving runModel for each decay: {e}")
 


  sys.exit() 
