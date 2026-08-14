# Credit Risk Segmentation
## Predicting Probability of Default from a Statistical, Machine Learning, and Deep Learning Perspective

This project focused on predicting loan default risk using statistical, machine learning, and deep learning models. The objective was to solve a binary classification problem by applying appropriate data preprocessing, feature engineering, regularization techniques, and model evaluation metrics. In addition to comparing predictive models, I developed a credit scorecard based on a logistic regression model to translate the most influential factors into an interpretable risk scoring system.

## Project Report & Additional Documents
 - [View full report](STAT468_final_paper.pdf)


## Skills Demonstrated
- Data preprocessing
- Feature engineering
- Binary classification
- Logistic regression
- Machine learning
- Deep learning
- Model evaluation
- Scorecard development
- Model interpretation
- Data modeling
- Pipeline creation

## Use instructions from terminal
- Make sure that the command prompt is in the file current directory in the IDE 

- Open terminal in the IDE and enter the following command: "py -3.13 -m venv .venv"
What does this do? It creates a virtual environment which can be thought of as a kernel that holds for now a blank space specifically design for directories that have files which need to reproduce code 
that require specific versions of code packages. Here the virtual environment is called "venv".

- Activate the virtual environment by typing in the command prompt: (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& *venv direcrory path*\Scripts\Activate.ps1)
eg: For me it looks like (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\Users\~.venv\Scripts\Activate.ps1)
  
- What does this do? It activates the virtual environment meaning that it will now essentially be able to use the packages installed into the environment when needed for the current working directory.

- Now run the following command: "pip install -r requirements.txt"
What does this do? This downloads all of the packages from the requirements text file.

Important Note: Sometimes all the packages may not download from the text file. It is around 2-3 you may need to do so manually. Also, you need to install "ipykernel" to make the venv into a usable kernel for the Jupyter notebook.

Important Note: I have done the environment procedure above and I am able to run the scripts above with only manually needed to add the following packages:
* ipykernel
