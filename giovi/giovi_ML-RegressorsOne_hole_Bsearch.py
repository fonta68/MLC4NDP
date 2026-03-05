import os
import numpy as np
import pandas as pd
#import tensorflow as tf
import argparse
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
#import matplotlib.pyplot as plt

# Import per i regressori classici
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Import del wrapper per Keras tramite scikeras per grid search su LSTM
#from scikeras.wrappers import KerasRegressor



def elimina_blocchi_random(df, n, x):
    """
    Elimina blocchi completi di n righe consecutive da un DataFrame, distribuiti casualmente.
    Ogni blocco viene eliminato con probabilità x/100, garantendo che il numero totale di righe
    eliminate (sempre multipli di n) non superi x% delle righe totali.

    Il seed per la randomizzazione è impostato a 42.
    """
    np.random.seed(42)
    T = len(df)
    max_rimuovere = int(T * x / 100)
    blocks = [(i, i + n) for i in range(0, T) if i + n <= T]
    chosen_blocks = [block for block in blocks if np.random.rand() < (x / 100)]
    totale_rimosse = len(chosen_blocks) * n
    while totale_rimosse > max_rimuovere and chosen_blocks:
        idx = np.random.randint(len(chosen_blocks))
        chosen_blocks.pop(idx)
        totale_rimosse = len(chosen_blocks) * n
    indices_to_remove = []
    for start, end in chosen_blocks:
        indices_to_remove.extend(range(start, end))
    indices_to_remove = sorted(indices_to_remove)
    df_rimosso = df.iloc[indices_to_remove].copy()
    df_modificato = df.drop(df.index[indices_to_remove]).reset_index(drop=True)
    return df_modificato, df_rimosso


def create_windows(data, window_size, target_col, n_future):
    """
    Creates time windows for training a model.
    
    Args:
        data: numpy array with all features.
        window_size: number of past time steps used as input.
        target_col: index of the target column to predict.
        n_future: number of time steps ahead to predict.

    Returns:
        X: array of input sequences.
        y: single column array of future values at `n_future` steps ahead.
    """
    X, y = [], []
    for i in range(len(data) - (window_size + n_future - 1)):
        X.append(data[i:i + window_size])  # Past window_size observations
        y.append(data[i + window_size + n_future - 1, target_col])  # Single value `n_future` steps ahead
    return np.array(X), np.array(y)  # `y` is now a 1D array




def get_fold(data: pd.DataFrame, i: int, k: int):
    """
    Restituisce il training set e il test set per la i-esima fold in una k-fold cross-validation.

    Parametri:
      data (pd.DataFrame): il dataset completo.
      i (int): indice della fold (da 0 a k-1).
      k (int): numero totale di fold.

    Ritorna:
      tuple(pd.DataFrame, pd.DataFrame): (df_train, df_test)

    Nota:
      La porzione di test è pari a 1/k del dataset (utilizzando slicing semplice).
    """
    n = len(data)
    fold_size = n // k
    start_index = i * fold_size
    # Per l'ultima fold includiamo eventuali dati residui
    end_index = n if i == k - 1 else (i + 1) * fold_size

    df_test = data.iloc[start_index:end_index]
    df_train = pd.concat([data.iloc[:start_index], data.iloc[end_index:]], ignore_index=True)
    return df_train, df_test



if __name__ == '__main__':
    # Parsing degli argomenti (senza epochs e batch_size)
    parser = argparse.ArgumentParser(description="Temperature Forecasting Script")
    parser.add_argument('--input_file', type=str, required=True, help="Nome del file CSV di input")
    #parser.add_argument('--output_dir', type=str, required=True, help="Cartella di output per salvare file e grafici")
    parser.add_argument('--window_size', type=int, default=12, help="Dimensione della finestra temporale (default: 12)")
    parser.add_argument('--n_future', type=int, default=5, help="Numero di valori futuri da prevedere (default: 5)")
    parser.add_argument('--time_column', type=str, default=None, help="Nome della colonna tempo da escludere")
    parser.add_argument('--hole_dim', type=int, default=0, help="Dimensione dei blocchi da eliminare")
    parser.add_argument('--percentage_hole', type=float, default=0, help="Percentuale di punti persi")
    parser.add_argument('--test_only', type=bool, default=0, help="Se i buchi devono essere applicati solo sul test")
    parser.add_argument('--k_fold', type=int, default=0, help="Fold sul quale fare gli esperimenti (da 0 a 4)")
    args = parser.parse_args()

    
	# Si estrae il nome del file
    f_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    
    #os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f_name, exist_ok=True)

    # Carica i dati
    data = pd.read_csv(args.input_file)
    
    #data.iloc[:, 1:] = data.iloc[:, 1:].applymap(lambda x: round(x, 4))
    
    
    
    # Divisione in training (80%) e test (20%)
    split_index = int(len(data) * 0.8)
    df_train_data = data.iloc[:split_index]
    df_test_data = data.iloc[split_index:]

    # df_train, df_test = get_fold(data, args.k_fold, 5)


    if args.hole_dim > 0:
        df_test_data, _ = elimina_blocchi_random(df_test_data, args.hole_dim, args.percentage_hole)
        if not args.test_only:
            df_train_data, _ = elimina_blocchi_random(df_train_data, args.hole_dim, args.percentage_hole)

    if args.time_column and args.time_column in data.columns:
        df_train_data = df_train_data.drop(columns=[args.time_column])
        df_test_data = df_test_data.drop(columns=[args.time_column])

    train_data = df_train_data.to_numpy()
    test_data = df_test_data.to_numpy()

    # Colonna target: 't_int'
    target_col = df_train_data.columns.get_loc('t_int')

    # Creazione delle finestre temporali
    X_train, y_train = create_windows(train_data, args.window_size, target_col, args.n_future)
    X_test, y_test = create_windows(test_data, args.window_size, target_col, args.n_future)

    # Reshape X to fit the model
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    n_obs = y_test.shape[0]

    

    # Lista per il summary finale: accumulerà le metriche per ogni metodo e per ogni orizzonte
    final_summary_list = []

    
    

    #############################################
    # BAYESIAN SEARCH PER I REGRESSORI CLASSICI
    #############################################
    # Appiattimento delle finestre temporali: da (n_samples, window_size, n_features) a (n_samples, window_size * n_features)
    #X_train_flat = X_train.reshape((X_train.shape[0], -1))
    #X_test_flat = X_test.reshape((X_test.shape[0], -1))

    # Definizione dei regressori (incluso Ridge)
    regressors = {
        "LRR": LinearRegression(),
        "RgR": Ridge(),
        "RFR": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "GBR": GradientBoostingRegressor(random_state=42),
        "ETR": ExtraTreesRegressor(random_state=42),
        "KNR": KNeighborsRegressor()
    }

    # Definizione dei dizionari degli iperparametri per la BAYES search
    search_spaces = {
    "LRR": {},  # No hyperparams to search for basic LinearRegression
    "RgR": {
        # Example of searching alpha over a log-scaled range
        "alpha": Real(1e-10, 10, prior='log-uniform')
    },
    "RFR": {
        # Example: integer range for n_estimators, plus a few discrete depths
        "n_estimators": Integer(100, 1000, prior='uniform'),
        "max_depth": Categorical([None, 5, 10, 20])
    },
    "SVR": {
        # Because you're using MultiOutputRegressor(SVR(...)),
        # prefix the parameter names with 'estimator__'
        "C": Real(1e-4, 1e2, prior='log-uniform'),
        "epsilon": Real(1e-4, 10, prior='log-uniform')
    },
    "GBR": {
        "n_estimators": Integer(100, 1000, prior='uniform'),
        "learning_rate": Real(1e-3, 0.5, prior='log-uniform')
    },
    "ETR": {
        "n_estimators": Integer(100, 1000, prior='uniform'),
        "max_depth": Categorical([None, 5, 10, 20])
    },
    "KNR": {
        "n_neighbors": Integer(1, 15, prior='uniform')
    }
}
    
    #nome file best parameters
    file_name_best = (f"pars_{f_name}_{args.window_size}win_{args.k_fold}k_{args.n_future}fut_"
                         f"{args.hole_dim}hd_{args.percentage_hole}hp_{args.test_only}tso.csv")
    f_bestpars = open(file_name_best, 'w')
        

    for name, reg in regressors.items():
        print(f"Bayesian search for {name}")
        if search_spaces[name]:  
            # If there are hyperparameters to tune:
            bayes_search = BayesSearchCV(
                estimator=reg,
                search_spaces=search_spaces[name],
                n_iter=25,  # number of parameter sets to try
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=3,
                random_state=42
            )
            bayes_search.fit(X_train_flat, y_train)
            best_reg = bayes_search.best_estimator_
            f_bestpars.write(f"Best parameters for {name}: {bayes_search.best_params_}\n")
        else:
            # If no hyperparameters to tune:
            reg.fit(X_train_flat, y_train)
            best_reg = reg

        predictions_reg = best_reg.predict(X_test_flat)
        # Ensure predictions_reg is 1D (to avoid shape mismatch errors)
        predictions_reg = predictions_reg.reshape(-1, 1)
        # rounding 
        predictions_reg[:, 1:] = np.round(predictions_reg[:, 1:], 4)
        
        results_reg = pd.DataFrame(np.hstack([y_test.reshape(-1, 1), predictions_reg]),
                               columns=["Act", "Pre"])
        #print(f"Best parameters for {name}: {bayes_search.best_params_}")

        # metrics computation        
        abs_error = np.abs(y_test - predictions_reg.flatten())  # Ensure correct shape
        mae_mean = np.mean(abs_error)
        mae_std = np.std(abs_error)
        mae_max = np.max(abs_error)
        count_over_0_5 = np.sum(abs_error > 0.5)
        count_over_1_0 = np.sum(abs_error > 1.0)
        error_perc_over_0_5 = (count_over_0_5 / len(y_test)) * 100
        error_perc_over_1_0 = (count_over_1_0 / len(y_test)) * 100

        metrics_df = pd.DataFrame({
            "MAE_mean": [mae_mean],
            "MAE_max": [mae_max],
            "MAE_std": [mae_std],
            "Error>0.5 (%)": [error_perc_over_0_5],
            "Error>1.0 (%)": [error_perc_over_1_0]
        })

        #print(metrics_df)

        file_name_reg = (f"pred_{f_name}_{name}_{args.window_size}win_{args.k_fold}k_{args.n_future}fut_"
                         f"{args.hole_dim}hd_{args.percentage_hole}hp_{args.test_only}tso.csv")
        output_file_reg = os.path.join(f_name, file_name_reg) # WARN: era args.output_dir 
        with open(output_file_reg, 'w', newline='') as f:
            results_reg.to_csv(f, index=False, float_format="%.4f")
            #f.write("Aggregated Metrics per Prediction Column (transposed)\n")
            #metrics_df.to_csv(f)
        print(f"{name} -> File salvato in: {output_file_reg}")
        

        # Aggiungo le metriche per ciascun orizzonte al summary finale
        #for i in range(args.n_future):
        final_summary_list.append({
            "Method": name,
            "Horizon": f"t",
            "MAE_mean": mae_mean,
            "MAE_max": mae_max,
            "MAE_std": mae_std,
            "Error>0.5 (%)": error_perc_over_0_5,
            "Error>1.0 (%)": error_perc_over_1_0
        })

# f_bestpars.close()
		
    # Creazione del file finale di summary utilizzando final_summary_list
    final_summary_df = pd.DataFrame(final_summary_list)
    final_summary_file = os.path.join(
        f_name,   # WARN: era args.output_dir 
        f"final_summary_{args.window_size}window_{args.k_fold}k_fold_{args.n_future}future_"
        f"{args.hole_dim}holes_{args.percentage_hole}perc_{args.test_only}_testOnly_{n_obs}samples.csv"
    )
#    final_summary_df.to_csv(final_summary_file, index=False)
#    print(f"File finale con il riassunto delle metriche salvato in: {final_summary_file}")
