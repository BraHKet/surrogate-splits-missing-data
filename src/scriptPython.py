"""
The UCI Heart Disease dataset is a widely used dataset in the field of machine learning and data science for predicting heart disease. This dataset is part of the UCI Machine Learning Repository and has been utilized in numerous research studies and projects.
Overview
The dataset contains information about patients, including various medical attributes that are used to predict the presence of heart disease. The dataset is a combination of data from five different sources: Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog (Heart) Data Set. In total, the dataset comprises 1190 instances with 11 common features.
Features
The dataset includes the following features:
•	age: Age of the patient in years
•	sex: Gender of the patient (1 = male, 0 = female)
•	cp: Chest pain type (4 values)
•	trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
•	chol: Serum cholesterol in mg/dl
•	fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
•	restecg: Resting electrocardiographic results (values 0, 1, 2)
•	thalach: Maximum heart rate achieved
•	exang: Exercise-induced angina (1 = yes, 0 = no)
•	oldpeak: ST depression induced by exercise relative to rest
•	slope: The slope of the peak exercise ST segment
•	ca: Number of major vessels (0-3) colored by fluoroscopy
•	thal: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
Usage
The dataset is commonly used for training and evaluating machine learning models to predict heart disease. Various classifiers such as Random Forest, SVM, Logistic Regression, KNN, and Decision Tree have been applied to this dataset1.

The UCI Heart Disease dataset is a valuable resource for researchers and practitioners in the field of machine learning and healthcare. It provides a comprehensive set of features that can be used to develop predictive models for heart disease, contributing to better diagnosis and treatment planning2.
Altre informazioni:
1.	https://github.com/harinda0/Heart-Disease-Prediction-ML/blob/main/Final%20Project.ipynb
2.	https://github.com/rikhuijzer/heart-disease-dataset
- - - - - - - - -
https://gts.ai/dataset-download/uci-heart-disease-data/  il dataset
Riporta tutte le informazioni compresa la variabile target:
Num: presence or absence of heart disease

https://archive.ics.uci.edu/dataset/45/heart+disease  il dataset

"""





import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter

def insert_missing_values_by_percentage(df, target_col, missing_percentage):
    """
    Inserts a specified percentage of missing values (NaN) randomly into the feature columns of a DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        target_col (str): The name of the target column to exclude from modifications.
        missing_percentage (float): The percentage of values to replace with NaN (e.g., 5 for 5%).

    Returns:
        pd.DataFrame: A new DataFrame with the specified percentage of missing values.
    """
    # Creare una copia per non modificare il DataFrame originale
    df_missing = df.copy()
    
    # Separare le feature dal target
    features = df_missing.drop(columns=[target_col])
    
    # Calcolare il numero totale di elementi nelle feature
    n_rows, n_cols = features.shape
    total_elements = n_rows * n_cols
    
    # Calcolare il numero di valori da rendere mancanti
    n_missing = int(total_elements * (missing_percentage / 100))
    
    if n_missing == 0:
        print(f"Attenzione: Con {missing_percentage}%, nessun valore verrà reso mancante. Aumenta la percentuale o la dimensione dei dati.")
        return df_missing
    
    # Creare una maschera booleana (tutti False all'inizio)
    mask = np.full(features.shape, False)
    
    # Scegliere 'n_missing' indici piatti (da 0 a total_elements-1) in modo casuale e unico
    flat_indices = np.random.choice(total_elements, n_missing, replace=False)
    
    # Convertire gli indici piatti in indici (riga, colonna) e impostare True nella maschera
    row_indices, col_indices = np.unravel_index(flat_indices, features.shape)
    mask[row_indices, col_indices] = True
    
    # Applicare la maschera al DataFrame delle feature per inserire i NaN
    features_with_nan = features.where(~mask, np.nan)
    
    # Riunire le feature modificate con la colonna target originale
    df_missing[features.columns] = features_with_nan
    
    return df_missing



def train_rpart_with_surrogates(df, target_col, test_size=0.3, random_state=42, 
                              cp=0.01, minsplit=20, maxdepth=10, usesurrogate=2, maxsurrogate=5):
    """
    Train an rpart decision tree with surrogate splits for missing data
    
    Parameters:
    - df: pandas DataFrame
    - target_col: string, name of target column
    - test_size: float, proportion for validation set
    - random_state: int, random seed for reproducibility
    - cp: complexity parameter
    - minsplit: minimum number of observations in a node to split
    - maxdepth: maximum depth of the tree
    - usesurrogate: how to use surrogates (0=no, 1=yes, 2=use surrogates for missing data)
    - maxsurrogate: the number of surrogate splits retained in the output
    
    Returns:
    - trained_model: R rpart model object
    - predictions: predictions on test set
    - accuracy: accuracy on test set
    """
    
    print("---------------------------")
    print("train_rpart_with_surrogates")
    print("---------------------------")
    print("usesurrogate = " + str(usesurrogate))
    print("maxsurrogate = " + str(maxsurrogate) + "\n")
    
    # Import R packages
    rpart = importr('rpart')
    base = importr('base')
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert to R objects using context manager and assign to R global environment
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["r_X"] = robjects.conversion.py2rpy(X)
        robjects.globalenv["r_y"] = robjects.conversion.py2rpy(y)
    

    # Split data in R
    r_code = f"""
    set.seed({random_state})
    
    # Create combined dataframe
    model_data <- cbind(r_X, target = r_y)
    model_data$target <- as.factor(model_data$target)
    
    # Split into training and validation
    n <- nrow(model_data)
    train_size <- floor({1-test_size} * n)
    train_indices <- sample(seq_len(n), size = train_size)
    
    train_data <- model_data[train_indices, ]
    test_data <- model_data[-train_indices, ]
    
    # Remove row names to avoid issues
    rownames(train_data) <- NULL
    rownames(test_data) <- NULL
    """
    robjects.r(r_code)
    
    # Train rpart model with surrogate splits
    # https://www.rdocumentation.org/packages/rpart/versions/4.1.24/topics/rpart.control
    r_code_train = f"""
    # Train rpart model with surrogate splits enabled
    rpart_model <- rpart(
        target ~ ., 
        data = train_data,
        method = "class",
        control = rpart.control(
            minsplit = {minsplit},
            minbucket = round({minsplit}/3),
            cp = {cp},
            maxdepth = {maxdepth},
            usesurrogate = {usesurrogate},  # Key parameter for surrogate splits
            maxsurrogate = {maxsurrogate},  # Maximum number of surrogate splits to retain
            xval = 10,                      # Number of cross-validations
            surrogatestyle = 1              # 1=percent correct
        ),
        parms = list(
            split = "gini"                  # Use gini index for splitting
        )
    )
    
    # Print model summary
    print("RPart Model Summary:")
    print(summary(rpart_model))
    
    # Make predictions on test set
    predictions <- predict(rpart_model, test_data, type = "class")
    
    # Calculate accuracy
    accuracy <- mean(predictions == test_data$target, na.rm = TRUE)
    
    # Get confusion matrix
    conf_matrix <- table(Predicted = predictions, Actual = test_data$target)
    
    # Get the number of surrogate splits actually used
    frame <- rpart_model$frame
    uses_surrogate <- sum(frame$ncompete > 0 | frame$nsurrogate > 0)
    """
    
    robjects.r(r_code_train)
    
    # Get results back to Python
    rpart_model = robjects.r['rpart_model']
    predictions = robjects.r['predictions']
    accuracy = robjects.r['accuracy'][0]
    conf_matrix = robjects.r['conf_matrix']
    uses_surrogate = robjects.r['uses_surrogate'][0]
    
    print(f"\nRPart Decision Tree trained successfully!")
    print(f"Uses surrogate splits: {uses_surrogate} nodes have surrogate splits")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:")
    print(conf_matrix)
    
    return rpart_model, predictions, accuracy



# next function not used
def train_rpart_advanced_surrogates(df, target_col, test_size=0.3, random_state=42,
                                  surrogate_params=None):
    """
    Advanced version with detailed control over surrogate splits
    """
    
    print("-------------------------------")
    print("train_rpart_advanced_surrogates")
    print("-------------------------------")

    rpart = importr('rpart')
    
    # Default surrogate parameters
    if surrogate_params is None:
        surrogate_params = {
            'usesurrogate': 2,
            'maxsurrogate': 5,
            'surrogatestyle': 1
        }
    
    # Convert entire dataframe to R using context manager
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["r_df"] = robjects.conversion.py2rpy(df)
    
    r_code = f"""
    set.seed({random_state})
    
    # Prepare data
    model_data <- r_df
    model_data${target_col} <- as.factor(model_data${target_col})
    
    # Split data
    n <- nrow(model_data)
    train_size <- floor({1-test_size} * n)
    train_indices <- sample(seq_len(n), size = train_size)
    
    train_data <- model_data[train_indices, ]
    test_data <- model_data[-train_indices, ]
    
    # Train rpart with detailed surrogate control
    rpart_model <- rpart(
        {target_col} ~ .,
        data = train_data,
        method = "class",
        control = rpart.control(
            minsplit = 20,
            minbucket = 7,
            cp = 0.01,
            maxdepth = 10,
            usesurrogate = {surrogate_params['usesurrogate']},
            maxsurrogate = {surrogate_params['maxsurrogate']},
            surrogatestyle = {surrogate_params['surrogatestyle']},
            xval = 10
        )
    )
    
    # Make predictions
    predictions <- predict(rpart_model, test_data, type = "class")
    accuracy <- mean(predictions == test_data${target_col}, na.rm = TRUE)
    
    # Get detailed information about surrogate usage
    frame <- rpart_model$frame
    total_nodes <- nrow(frame)
    nodes_with_surrogates <- sum(frame$nsurrogate > 0)
    """
    
    robjects.r(r_code)
    
    rpart_model = robjects.r['rpart_model']
    predictions = robjects.r['predictions']
    accuracy = robjects.r['accuracy'][0]
    total_nodes = robjects.r['total_nodes'][0]
    nodes_with_surrogates = robjects.r['nodes_with_surrogates'][0]
    
    print(f"Advanced RPart with surrogate splits trained!")
    print(f"Total nodes: {total_nodes}")
    print(f"Nodes using surrogate splits: {nodes_with_surrogates}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return rpart_model, predictions, accuracy



def get_surrogate_info(model):
    """Get information about surrogate splits in the model"""
    
    print("------------------")
    print("get_surrogate_info")
    print("------------------")

    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["rpart_model"] = robjects.conversion.py2rpy(model)

    #robjects.r['rpart_model'] = model
    
    r_code = """
    # Get frame information
    frame_df <- as.data.frame(rpart_model$frame)
    frame_df$node_name <- rownames(rpart_model$frame)
    
    # Count nodes with surrogate splits
    surrogate_nodes <- frame_df[frame_df$nsurrogate > 0, c("node_name", "ncompete", "nsurrogate")]
    """
    
    robjects.r(r_code)
    
    # Convert surrogate_nodes back to pandas DataFrame using context manager
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        surrogate_nodes = robjects.conversion.rpy2py(robjects.r['surrogate_nodes'])
    
    print("Nodes using surrogate splits:")
    if len(surrogate_nodes) > 0:
        print(surrogate_nodes)
    else:
        print("No surrogate splits used in this tree")
    
    return surrogate_nodes




def make_rpart_predictions_with_surrogates(model, new_data):
    """
    Make predictions using the rpart model with surrogate splits
    """

    print("--------------------------------------")
    print("make_rpart_predictions_with_surrogates")
    print("--------------------------------------")

    # Convert new_data to R using context manager
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        r_new_data = robjects.conversion.py2rpy(new_data)
    
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["rpart_model"] = robjects.conversion.py2rpy(model)
        robjects.globalenv["new_data"] = robjects.conversion.py2rpy(r_new_data)
   
    # Surrogate splits will be automatically used for missing values during prediction
    predictions = robjects.r('predict(rpart_model, new_data, type = "class")')
    
    # Convert predictions back to Python using context manager
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        python_predictions = robjects.conversion.rpy2py(predictions)
    
    return python_predictions




def get_variable_importance(model):
    """Get variable importance from the model"""

    print("-----------------------")
    print("get_variable_importance")
    print("-----------------------")

    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        robjects.globalenv["rpart_model"] = robjects.conversion.py2rpy(model)
#   robjects.r['rpart_model'] = model
    
    r_code = """
    # Get variable importance
    var_imp <- rpart_model$variable.importance
    var_imp_df <- data.frame(
        Variable = names(var_imp),
        Importance = as.numeric(var_imp)
    )
    var_imp_df <- var_imp_df[order(-var_imp_df$Importance), ]
    """
    
    robjects.r(r_code)
    
    # Convert variable importance back to pandas DataFrame
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        var_imp_df = robjects.conversion.rpy2py(robjects.r['var_imp_df'])
    
    return var_imp_df

def plot_rpart_tree(model, filename=None):
    """Plot the decision tree and save to file"""

    print("---------------")
    print("plot_rpart_tree")
    print("---------------")

    try:
        rpart_plot = importr('rpart.plot')
        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            robjects.globalenv["rpart_model"] = robjects.conversion.py2rpy(model)
               
        if filename:
            r_code = f"""
            png('{filename}', width=800, height=600)
            rpart.plot(rpart_model, main="Decision Tree with Surrogate Splits")
            dev.off()
            """
        else:
            r_code = """
            rpart.plot(rpart_model, main="Decision Tree with Surrogate Splits")
            """
        
        robjects.r(r_code)
        print(f"Tree plot saved to {filename}" if filename else "Tree displayed")
        
    except Exception as e:
        print(f"Could not plot tree: {e}")
        print("Please install rpart.plot package in R: install.packages('rpart.plot')")




# Example usage

print("-----")
print("main")
print("-----")

if __name__ == "__main__":
    
    datasetnumber = 1  # 0: synthetic data, 1: UCI Heart Disease dataset , 2: Fisher iris
    
    match datasetnumber:
        case 0:
            # Create sample data with missing values
            np.random.seed(42)
            n_samples = 500
    
            sample_df = pd.DataFrame({
                'age': np.random.randint(20, 70, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'education': np.random.randint(1, 5, n_samples),
                'credit_score': np.random.normal(650, 100, n_samples),
                'target': np.random.randint(0, 2, n_samples)
            })
        
            # Introduce missing values
            missing_income = np.random.random(n_samples) < 0.15
            missing_credit = np.random.random(n_samples) < 0.15
            sample_df.loc[missing_income, 'income'] = np.nan
            sample_df.loc[missing_credit, 'credit_score'] = np.nan
 
        case 1:   # use the UCI Heart Disease dataset 
        # Load the dataset
            url = "https://raw.githubusercontent.com/rikhuijzer/heart-disease-dataset/main/heart-disease-dataset.csv"
            sample_df = pd.read_csv(url)

            # --- 2. INTRODUCIAMO I MISSING SU TUTTO IL DATASET ---
            MISSING_PERCENTAGE = 90.0
            print("="*60)
            print(f"FASE 1: Introduzione del {MISSING_PERCENTAGE}% di valori mancanti sull'intero dataset...")
            print("="*60)
    
            if MISSING_PERCENTAGE > 0:
                df_with_missing = insert_missing_values_by_percentage(
                df=sample_df,
                target_col='target',
                missing_percentage=MISSING_PERCENTAGE
            )
            else:
                df_with_missing = sample_df.copy()

            sample_df = df_with_missing
            # ----------------


            n_samples = len(sample_df)  # sample_df.shape[0]

            # Display the first few rows of the dataset
            print(sample_df.head())

            # Preprocess the dataset (e.g., handling missing values, encoding categorical variables)
            # Example: Fill missing values with the mean of the column
            #sample_df.fillna(data.mean(), inplace=True)

        case 2:   # Fisher's iris
            from sklearn.datasets import load_iris

            # Load the Iris dataset
            iris = load_iris()

            # Convert to Pandas DataFrame
            sample_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

            # Add the target column
            sample_df['target'] = iris.target  # [0, 1, 2] correspond to ['setosa', 'versicolor', 'virginica']

            # Display the first few rows
            print(sample_df.head())
            
            # Choose one of the three species as positive, the other two as negative
            positiveclass = 0
            y = sample_df['target']  # view!!
            y[y == positiveclass] = 99
            y[y != 99] = 0  # negative
            y[y == 99] = 1  # positive
            print(sample_df)

        case _:
            print("DATASET NOT AVAILABLE")
            exit()
            
    ## Example: Encode categorical variables
    #sample_df['sex'] = data['sex'].map({0: 'female', 1: 'male'})   
    
    print("DataFrame shape:", sample_df.shape)
    print("Missing values per column:")
    print(sample_df.isnull().sum())
    
    # Train rpart with or without surrogate splits
    
    usesurrogate = 1 # 0=no, 1=yes, 2=use surrogates for missing data (0 or 2)
    if usesurrogate == 0:
        maxsurrogate = 0
    else:
        maxsurrogate = 5
        
    rpart_model, predictions, accuracy = train_rpart_with_surrogates(
        df=sample_df,
        target_col='target',
        test_size=0.3,
        random_state=42,
        cp=0.01,
        minsplit=20,
        maxdepth=5,
        usesurrogate=usesurrogate,
        maxsurrogate=maxsurrogate
    )
    
    # Get surrogate information
    surrogate_info = get_surrogate_info(rpart_model)
    
    # Get variable importance
    var_importance = get_variable_importance(rpart_model)
    print("\nVariable Importance:")
    print(var_importance)
    
    '''
    # Make predictions on new data with missing values
    new_data = sample_df.drop('target', axis=1).iloc[:5].copy()
    new_data.loc[0, 'income'] = np.nan
    new_data.loc[1, 'credit_score'] = np.nan
    
    print("\nNew data with missing values:")
    print(new_data[['age', 'income', 'credit_score']])
    
    new_predictions = make_rpart_predictions_with_surrogates(rpart_model, new_data)
    print("Predictions on new data (using surrogate splits):")
    print(new_predictions)
    '''
    
    # Plot the tree
    plot_rpart_tree(rpart_model, "decision_tree.png") # save to file
    
    
    
    
    
    '''
    R INSTALLATION
    
https://cran.mirror.garr.it/CRAN/
Scelgo Ubuntu

# update indices
sudo apt update -qq
# install two helper packages we need
sudo apt install --no-install-recommends software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the repo from CRAN -- lsb_release adjusts to 'noble' or 'jammy' or ... as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# install R itself
sudo apt install --no-install-recommends r-base



Nella console, chiamata da shell con il comando: R

install.packages("rpart")
install.packages('rpart.plot')

q()


conda create -p ./missing_env python pandas rpy2 scikit-learn


    '''
    
