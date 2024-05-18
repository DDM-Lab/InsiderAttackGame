# InsiderAttackGame

Aggarwal, P., Cranford, E.A., Tambe, M., Lebiere, C., Gonzalez, C. (2023). Deceptive Signaling: Understanding Human Behavior Against Signaling Algorithms. In: Bao, T., Tambe, M., Wang, C. (eds) Cyber Deception. Advances in Information Security, vol 89. Springer, Cham. https://doi-org.cmu.idm.oclc.org/10.1007/978-3-031-16613-6_5

## Data files

The human experiment dataset are available on OSF: https://osf.io/c7ntu/?view_only=0e6261b5e818440495d9917044611758

For this particular implementation we are using:
    - 2022-MURIBookChapter-FullData-IAG.csv
    - AllConditions_humanData_dictionary.txt (dictionary with variables information)

max_decays_IAG.csv was generated after running Model-Fitting for each model and it contain the best decay for each participant.



## IBL model

insiderAttackGame.py contains the implementation of the IBL model for model-tracing.

confusionMatric.py contains functions to generate confusion matrixes for both models in different settings.

runFitting.py has the developped of the Model-Fitting implementation according to the IBL model in the insiderAttackGame.py.