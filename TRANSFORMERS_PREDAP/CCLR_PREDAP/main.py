import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import time

matplotlib.use('Agg')  # non-interactive backend (no GUI)
from multiprocessing import freeze_support
from sklearn.preprocessing import MinMaxScaler

from CCLR_PREDAP.src.lmlr import (
    smoother, plot_example, ploter,
    get_top_correlations_blog, compute_vif,compute_vif_matrix, filter_VIF, compute_vif_matrix_gpu,
    models_training, metrics_calculation, evaluation_metrics_MAPE, 
    evaluation_metrics_RMSE, evaluation_metrics_Ftest
)

from CCLR_PREDAP.src.gcausal import (
    smoother, stationate, lag_plots,
    kpss_test, splitter, select_p, granger_causation_matrix,granger_causation_matrix_parallel, min_max_scale,
    fit_var_model, recommend_lag_order, select_causal_features
)

from CCLR_PREDAP.src.dl import (
    add_temprality, plot_train_test, create_model_gru,
    create_model_lstm, create_model_bilstm, create_model_enc_dec,
    create_model_enc_dec_cnn, create_model_vector_output, create_model_multi_head_cnn_lstm,
    split_sequence, fit_model, prediction, inverse_transform,
    evaluate_forecast,
    plot_train_test, plt_model

)   




    # LMLR PHASE 
    # --- Load and preprocess data ---
    #HYPERPARAMETERS

def CCLR_pipeline(
        data_paths, 
        CODES_LIST, 
        CORRELATION_THRESHOLD = 0.90, 
        ITERATIONS_MAX = 250, 
        VIF_THRESHOLD = 20.0, 
        MAX_ITERS_MODEL = 60, 
        LAGS_LIST = [1,3,7,14,30, 60, 182, 365], 
        BEST_FEATURES_PATH = f'../data/best_features/BEST_features_NOSMOOTH_'
    ):

    for data_path in data_paths:
        # RAW DATA --------------------------------------------------------------
        # Get the directory of the current script
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root, then into data folder
        #data_path = os.path.join(script_dir, 'data', 'synthetic_timeseries.csv')
        #data_name = 'FINAL_diagnostics_CAT'
        #CODES_LIST = dict_data_names[data_name]
        
        #Check if dataframe is in csv or parquet format
        if data_path.endswith('.csv'):
            data_name = os.path.basename(data_path)
            data_path = '../data/FINAL_DB/' + data_name
            df = pd.read_csv(data_path, index_col=0)
        else:
            data_name = os.path.basename(data_path)
            data_path = '../data/FINAL_DB/' + data_name 
            df = pd.read_parquet(data_path)

        df['timestamp'] = df.index
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp('2007-01-01')
        #df = df[df['timestamp'] >= cutoff].reset_index(drop=True)  # Subset the DataFrame
        df = df.set_index('timestamp')
        # SMOOTHING DATA --------------------------------------------------------
        '''smoothed = smoother(df, WINDOW_SIZE)
        plot_example(smoothed, "SMOOTHED (example 10 ts)")'''

        # SCALED DATA ----------------------------------------------------------
        # Min-Max Scaling
        df_scaled = MinMaxScaler().fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        #df_scaled = min_max_scale(df) # normalize df
        plot_example(df_scaled, " SCALED DATA (example 10 ts)")


        # FILTER VARIABLES ---------------------------------------------------
        # Make a copy of the smoothed dataframe
        dataframe = df_scaled.copy()

        # Compute top correlations above a threshold
        df_correlations = get_top_correlations_blog(dataframe, threshold=CORRELATION_THRESHOLD)
        print(df_correlations)

        # Extract correlated variable names from both levels of the MultiIndex
        correlated_vars = list(set(df_correlations.index.get_level_values(0).tolist() + 
                                df_correlations.index.get_level_values(1).tolist()))

        # Compute variables not involved in any strong correlation
        non_correlated_vars = list(set(dataframe.columns) - set(correlated_vars))

        # variables que són subjecte de ser eliminades
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        if f"vif_analysis1_{data_name}.xlsx" not in os.listdir('plots') or f"redundant_variables_{data_name}.xlsx" not in os.listdir('plots'):
            print(">>> VIF analysis not found. Computing VIF ...")
            # VIF ---------------------------------------------------
            # compute VIF de les variables no correlacionades
            #vif = compute_vif(correlated_vars, dataframe).sort_values('VIF', ascending=False)
            start_time_gpu = time.time()
            #vif = compute_vif_matrix_gpu(correlated_vars, dataframe).sort_values('VIF', ascending=False)
            end_time_gpu = time.time()
            #print("VIF GPU: Done")
            #print(f"VIF computation time (GPU): {end_time_gpu - start_time_gpu} seconds")
            start_time_cpu = time.time()
            vif = compute_vif_matrix(non_correlated_vars, dataframe).sort_values('VIF', ascending=False)
            end_time_cpu = time.time()
            
            print(f"VIF computation time (CPU): {end_time_cpu - start_time_cpu} seconds")
            #print(f"VIF computation time (GPU): {end_time_gpu - start_time_gpu} seconds")
            pd.DataFrame(vif).to_excel(f"plots/vif_analysis_{data_name}.xlsx")
            print(vif)

            # filtrar el vif
            non_redundant_vars = filter_VIF(vif, dataframe, ITERATIONS_MAX, VIF_THRESHOLD)
            
            non_redundant_vars_dataframe = pd.DataFrame(non_redundant_vars, columns=['Redundant Variables'])
            non_redundant_vars_dataframe.to_excel(f"plots/redundant_variables_{data_name}.xlsx", index=False)
            #redundant_vars = []
            print(">>> Redundant variables to remove due to high VIF ... %s" % len(non_redundant_vars))

        else:
            print(">>> VIF analysis found. Loading redundant variables ...")
            non_redundant_vars_df = pd.read_excel(f"plots/redundant_variables_{data_name}.xlsx", engine='openpyxl')
            non_redundant_vars = non_redundant_vars_df['Redundant Variables'].tolist()

        #non_redundant_vars = list(set(dataframe.columns) - set(redundant_vars))
        for CODE in CODES_LIST:

            # BEST_FEATURES_PATH
            BEST_FEATURES_PATH = BEST_FEATURES_PATH + f"{CODE}.xlsx"


            dataframe = df_scaled.copy()
            # subset of dataframe with interesting vars
            print(">>> Initial number of variables in dataframe ... %s" % dataframe.shape[1])
            interesting_vars = list(set(non_redundant_vars))

            dataframe = dataframe[interesting_vars]
            dataframe[CODE] = df_scaled[CODE] # tornem a addherir la variable a predir

            print(">>> Final number of variables in dataframe ... %s" % dataframe.shape[1])
            non_scaled_df = dataframe.copy()

            print("... Scaling dataframe ===>")
            dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min() + 1e-8) # scaling again after filtering variables

            # MODELS CONSTRUCTION ------------------------------------------------------
            # sort absolute correlations between variables and cov19

            non_scaled_df = non_scaled_df.drop([CODE], axis=1)
            smot_corr = non_scaled_df.corrwith(df_scaled[CODE]).sort_values(ascending=False, key=abs)



            # A. initial model (on-time prediction) ----------------
            df_init = models_training(dataframe, CODE, smot_corr, MAX_ITERS_MODEL)


            # B. lagged models (1 to MAX_LAG days ahead) ----------------

            BEST = df_init[df_init["BEST_MODEL"] == "YES"]
            BEST["LAG"] = 0 

            for i in LAGS_LIST:
                dat_lag = dataframe.copy()
                # add lag
                dat_lag[CODE] = dat_lag[CODE].shift(-i)
                dat_lag = dat_lag.dropna(subset=[CODE])

                # OBTENIR EL MILLOR MODEL
                df_init = models_training(dat_lag, CODE, smot_corr, MAX_ITERS_MODEL, plt_models=False, plt_metrics=False, plt_selected_models=False, plt_best_model=False)
                best = df_init[df_init["BEST_MODEL"] == "YES"]
                best["LAG"] = i
                BEST = pd.concat([BEST,best]).reset_index(drop=True)

            print(BEST)

            # Save best features into an Excel file 
            BEST.to_excel(BEST_FEATURES_PATH, index=False, engine='openpyxl')

            #EVALUATION METRICS ANALYISIS ------------------------------------------------------
            evaluation_metrics_MAPE(BEST, code=CODE)
            evaluation_metrics_RMSE(BEST, code = CODE)
            evaluation_metrics_Ftest(BEST, code=CODE)

            # GCAUSAL PHASE 

            # Define our target series 
            target = [CODE] # Target variable

            # Define our predictor series
            # Load selected features from BEST_features_NOSMOOTH.xlsx
            best_features_df = pd.read_excel(BEST_FEATURES_PATH, engine='openpyxl')
            # Exclude the target column from predictors
            predictors = list(set([element for i in best_features_df['predictors'] for element in i.split(',')]))
            predictors = [col for col in predictors if col != CODE]

            ### 0.1. Stationarity (check)
            variables = [target+predictors]
            variables = [item for sublist in variables for item in sublist]
            lag_plots(df[variables]) # lag plots to visualize trends

            # KPSS test
            print(kpss_test(df[variables]))

            indexes = kpss_test(df[variables]).T[kpss_test(df[variables]).T['p-value'] < 0.05].index.tolist()
            print(indexes)

            # first order stationarity
            stationate_df = stationate(df[variables],indexes)
            lag_plots(stationate_df[variables[0:len(variables)]])

            print(kpss_test(stationate_df))
            indexes = kpss_test(stationate_df).T[kpss_test(stationate_df).T['p-value'] < 0.05].index.tolist()
            print(indexes)

            # second order stationarity
            stationate_df = stationate(stationate_df,indexes)
            lag_plots(stationate_df[variables[0:len(variables)]])

            print(kpss_test(stationate_df))


            ### 0.2. G-Causality test
            train_df, test_df = splitter(stationate_df)
            results_df, optimal_lags = select_p(train_df, LAGS_LIST)
            opt_lag = recommend_lag_order(optimal_lags=optimal_lags)# Optimal lag based on criteria from previous step


            var_result = fit_var_model(train_df, opt_lag)


if __name__ == "__main__":
    freeze_support()
    # SMOOTHING
    WINDOW_SIZE = 14
    # CORR
    CORRELATION_THRESHOLD = 0.90 # Above this value, variables are considered strongly correlated
    # VIF
    ITERATIONS_MAX = 250
    ITERATIONS = 0
    VIF_THRESHOLD = 20.0

    #MODELS 
    MAX_ITERS_MODEL = 60
    MAX_LAG = 30 

    LAGS_LIST = [1,3,7,14,30, 60, 182, 365]
    #data_names = ['FINAL_diagnostics_CAT', 'FINAL_diagnostics_RS', 'FINAL_diagnostics_UP']
    #data_names = ['FINAL_diagnostics_RS', 'FINAL_diagnostics_UP']
    #data_names = ['FINAL_diagnostics_UP']
    #data_names = ['finals_combined.csv']
    data_names = ['demand_diagnosis_joined.parquet']
    '''    dict_data_names = {
        'FINAL_diagnostics_CAT': ['total', 'Chapter10:J00-J99', 'Ch10:subch01:J00-J06','J00','Chapter18:R00-R99','M54'],
        'FINAL_diagnostics_RS': ['BARCELONAMETROPOLITANANORD_76', 'Chapter10:J00-J99_BARCELONAMETROPOLITANANORD_76', 'Ch10:subch01:J00-J06_BARCELONAMETROPOLITANANORD_76','J00_BARCELONAMETROPOLITANANORD_76','Chapter18:R00-R99_BARCELONAMETROPOLITANANORD_76','M54_BARCELONAMETROPOLITANANORD_76'],
        'FINAL_diagnostics_UP': ['00108', 'Chapter10:J00-J99_00108', 'Ch10:subch01:J00-J06_00108','J00_00108','Chapter18:R00-R99_00108','M54_00108']
    }'''
    #CODES_LIST = ['Ch10:subch10:J95-J95', 'Ch20:subch23:X30-X39', 'Ch21:subch13:Z68-Z68', 'Ch20:subch04:V10-V19', 'Ch08:subch05:H95-H95']
    #CODES_LIST = ['total', 'Chapter10:J00-J99', 'Ch10:subch01:J00-J06','J00','Chapter18:R00-R99','M54']
    CODES_LIST = ["DEMAND_demanda_SERVEI_CODI_INF",
                    
                    "DEMAND_demanda_SERVEI_CODI_MF",
                    "DEMAND_demanda_SERVEI_CODI_PED",
                    "DEMAND_demanda_SERVEI_CODI_URG",
                    "DEMAND_demanda_TIPUS_CLASS_9T",
                    "DEMAND_demanda_TIPUS_CLASS_C9C",
                    "DEMAND_demanda_TIPUS_CLASS_C9R",
                ]
    
    '''CODES_LIST = ["demanda__SERVEI_CODI__INF",
                "demanda__SERVEI_CODI__INFP",
                "demanda__SERVEI_CODI__MF",
                "demanda__SERVEI_CODI__PED",
                "demanda__SERVEI_CODI__URG",
                "demanda__TIPUS_CLASS__9T",
                "demanda__TIPUS_CLASS__C9C",
                "demanda__TIPUS_CLASS__C9R",
                ]'''
    for data_name in data_names:
        # RAW DATA --------------------------------------------------------------
        # Get the directory of the current script
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root, then into data folder
        #data_path = os.path.join(script_dir, 'data', 'synthetic_timeseries.csv')
        #data_name = 'FINAL_diagnostics_CAT'
        #CODES_LIST = dict_data_names[data_name]
        
        #Check if dataframe is in csv or parquet format
        if data_name.endswith('.csv'):
            data_path = '../data/FINAL_DB/' + data_name
            df = pd.read_csv(data_path, index_col=0)
        else:
            data_path = '../data/FINAL_DB/' + data_name 
            df = pd.read_parquet(data_path)

        df['timestamp'] = df.index
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = pd.Timestamp('2007-01-01')
        #df = df[df['timestamp'] >= cutoff].reset_index(drop=True)  # Subset the DataFrame
        df = df.set_index('timestamp')
        # SMOOTHING DATA --------------------------------------------------------
        '''smoothed = smoother(df, WINDOW_SIZE)
        plot_example(smoothed, "SMOOTHED (example 10 ts)")'''

        # SCALED DATA ----------------------------------------------------------
        # Min-Max Scaling
        df_scaled = MinMaxScaler().fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        #df_scaled = min_max_scale(df) # normalize df
        plot_example(df_scaled, " SCALED DATA (example 10 ts)")


        # FILTER VARIABLES ---------------------------------------------------
        # Make a copy of the smoothed dataframe
        dataframe = df_scaled.copy()

        # Compute top correlations above a threshold
        df_correlations = get_top_correlations_blog(dataframe, threshold=CORRELATION_THRESHOLD)
        print(df_correlations)

        # Extract correlated variable names from both levels of the MultiIndex
        correlated_vars = list(set(df_correlations.index.get_level_values(0).tolist() + 
                                df_correlations.index.get_level_values(1).tolist()))

        # Compute variables not involved in any strong correlation
        non_correlated_vars = list(set(dataframe.columns) - set(correlated_vars))

        # variables que són subjecte de ser eliminades
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        if f"vif_analysis1_{data_name}.xlsx" not in os.listdir('plots') or f"redundant_variables_{data_name}.xlsx" not in os.listdir('plots'):
            print(">>> VIF analysis not found. Computing VIF ...")
            # VIF ---------------------------------------------------
            # compute VIF de les variables no correlacionades
            #vif = compute_vif(correlated_vars, dataframe).sort_values('VIF', ascending=False)
            start_time_gpu = time.time()
            #vif = compute_vif_matrix_gpu(correlated_vars, dataframe).sort_values('VIF', ascending=False)
            end_time_gpu = time.time()
            #print("VIF GPU: Done")
            #print(f"VIF computation time (GPU): {end_time_gpu - start_time_gpu} seconds")
            start_time_cpu = time.time()
            vif = compute_vif_matrix(non_correlated_vars, dataframe).sort_values('VIF', ascending=False)
            end_time_cpu = time.time()
            
            print(f"VIF computation time (CPU): {end_time_cpu - start_time_cpu} seconds")
            #print(f"VIF computation time (GPU): {end_time_gpu - start_time_gpu} seconds")
            pd.DataFrame(vif).to_excel(f"plots/vif_analysis_{data_name}.xlsx")
            print(vif)

            # filtrar el vif
            non_redundant_vars = filter_VIF(vif, dataframe, ITERATIONS_MAX, VIF_THRESHOLD)
            
            non_redundant_vars_dataframe = pd.DataFrame(non_redundant_vars, columns=['Redundant Variables'])
            non_redundant_vars_dataframe.to_excel(f"plots/redundant_variables_{data_name}.xlsx", index=False)
            #redundant_vars = []
            print(">>> Redundant variables to remove due to high VIF ... %s" % len(non_redundant_vars))

        else:
            print(">>> VIF analysis found. Loading redundant variables ...")
            non_redundant_vars_df = pd.read_excel(f"plots/redundant_variables_{data_name}.xlsx", engine='openpyxl')
            non_redundant_vars = non_redundant_vars_df['Redundant Variables'].tolist()

        #non_redundant_vars = list(set(dataframe.columns) - set(redundant_vars))
        for CODE in CODES_LIST:

            # BEST_FEATURES_PATH
            BEST_FEATURES_PATH = f'BEST_features_NOSMOOTH_{CODE}.xlsx'


            dataframe = df_scaled.copy()
            # subset of dataframe with interesting vars
            print(">>> Initial number of variables in dataframe ... %s" % dataframe.shape[1])
            interesting_vars = list(set(non_redundant_vars))

            dataframe = dataframe[interesting_vars]
            dataframe[CODE] = df_scaled[CODE] # tornem a addherir la variable a predir

            print(">>> Final number of variables in dataframe ... %s" % dataframe.shape[1])
            non_scaled_df = dataframe.copy()

            print("... Scaling dataframe ===>")
            dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min() + 1e-8) # scaling again after filtering variables

            # MODELS CONSTRUCTION ------------------------------------------------------
            # sort absolute correlations between variables and cov19

            non_scaled_df = non_scaled_df.drop([CODE], axis=1)
            smot_corr = non_scaled_df.corrwith(df_scaled[CODE]).sort_values(ascending=False, key=abs)



            # A. initial model (on-time prediction) ----------------
            df_init = models_training(dataframe, CODE, smot_corr, MAX_ITERS_MODEL)


            # B. lagged models (1 to MAX_LAG days ahead) ----------------

            BEST = df_init[df_init["BEST_MODEL"] == "YES"]
            BEST["LAG"] = 0 

            for i in LAGS_LIST:
                dat_lag = dataframe.copy()
                # add lag
                dat_lag[CODE] = dat_lag[CODE].shift(-i)
                dat_lag = dat_lag.dropna(subset=[CODE])

                # OBTENIR EL MILLOR MODEL
                df_init = models_training(dat_lag, CODE, smot_corr, MAX_ITERS_MODEL, plt_models=False, plt_metrics=False, plt_selected_models=False, plt_best_model=False)
                best = df_init[df_init["BEST_MODEL"] == "YES"]
                best["LAG"] = i
                BEST = pd.concat([BEST,best]).reset_index(drop=True)

            print(BEST)

            # Save best features into an Excel file 
            BEST.to_excel(BEST_FEATURES_PATH, index=False, engine='openpyxl')

            #EVALUATION METRICS ANALYISIS ------------------------------------------------------
            evaluation_metrics_MAPE(BEST, code=CODE)
            evaluation_metrics_RMSE(BEST, code = CODE)
            evaluation_metrics_Ftest(BEST, code=CODE)

            # GCAUSAL PHASE 

            # Define our target series 
            target = [CODE] # Target variable

            # Define our predictor series
            # Load selected features from BEST_features_NOSMOOTH.xlsx
            best_features_df = pd.read_excel(BEST_FEATURES_PATH, engine='openpyxl')
            # Exclude the target column from predictors
            predictors = list(set([element for i in best_features_df['predictors'] for element in i.split(',')]))
            predictors = [col for col in predictors if col != CODE]

            ### 0.1. Stationarity (check)
            variables = [target+predictors]
            variables = [item for sublist in variables for item in sublist]
            lag_plots(df[variables]) # lag plots to visualize trends

            # KPSS test
            print(kpss_test(df[variables]))

            indexes = kpss_test(df[variables]).T[kpss_test(df[variables]).T['p-value'] < 0.05].index.tolist()
            print(indexes)

            # first order stationarity
            stationate_df = stationate(df[variables],indexes)
            lag_plots(stationate_df[variables[0:len(variables)]])

            print(kpss_test(stationate_df))
            indexes = kpss_test(stationate_df).T[kpss_test(stationate_df).T['p-value'] < 0.05].index.tolist()
            print(indexes)

            # second order stationarity
            stationate_df = stationate(stationate_df,indexes)
            lag_plots(stationate_df[variables[0:len(variables)]])

            print(kpss_test(stationate_df))


            ### 0.2. G-Causality test
            train_df, test_df = splitter(stationate_df)
            results_df, optimal_lags = select_p(train_df, LAGS_LIST)
            opt_lag = recommend_lag_order(optimal_lags=optimal_lags)# Optimal lag based on criteria from previous step


            var_result = fit_var_model(train_df, opt_lag)
            '''
            print(granger_causation_matrix(train_df, train_df.columns, p=1))
            print(granger_causation_matrix(train_df, train_df.columns, p=7))
            print(granger_causation_matrix(train_df, train_df.columns, p=30))'''
            
            '''
            causal_features_df = pd.DataFrame(columns=['LAG', 'features']) 

            if LAGS_LIST is not None:
                for i in LAGS_LIST:
                    #gmatrix = granger_causation_matrix(train_df, train_df.columns, p=i)
                    gmatrix = granger_causation_matrix(train_df, train_df.columns, p=i)
                    causal_features = select_causal_features(gmatrix, target_variable=CODE, significance_level=0.05)
                    causal_features_df = pd.concat([causal_features_df, pd.DataFrame({'LAG': i, 'features': [causal_features]})], ignore_index=True)

                causal_features_df.to_excel(f'causal_features_per_lag_{CODE}.xlsx', index=False, engine='openpyxl')
                
            else:
                #gmatrix = granger_causation_matrix(train_df, train_df.columns, p=opt_lag)
                gmatrix = granger_causation_matrix(train_df, train_df.columns, p=opt_lag)
                causal_features = select_causal_features(gmatrix, target_variable=CODE, significance_level=0.05)
                causal_features_df = pd.concat([causal_features_df, pd.DataFrame({'LAG': opt_lag, 'features': [causal_features]})], ignore_index=True)
                print("The selected causal features are:", causal_features)
            '''
            '''
            # DL PHASE

            objective = CODE 
            cc_predictors = causal_features # predictors based on G-causality results

            # df definition ..............................................
            cc_predictors.append(objective)
            variables_predictors = cc_predictors.copy()
            subdf = df[cc_predictors]

            # Add temporality (weekday and month) ........................
            TEMPORALITY = False
            if TEMPORALITY:
                subdf, cc_predictors = add_temprality(subdf, cc_predictors)

            # put objective to last column ..............................
            cc_predictors = [col for col in subdf.columns if col != objective]
            cc_predictors.append(objective)
            subdf = subdf[cc_predictors]
            print(subdf) 

            # train-test split ...........................................
            TRAIN_PERCENTAGE = 0.8
            train_size = int(len(subdf)*TRAIN_PERCENTAGE)
            train_dataset, test_dataset = subdf.iloc[5:train_size],subdf.iloc[train_size:]

            # Plot train and test data ...................................
            plot_train_test(train_dataset, test_dataset, objective)

            #scale ......................................................
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_dataset)
            scaled_test = scaler.transform(test_dataset)

            # get idx of objective variable
            col_idx = subdf.columns.get_loc(objective)

            # DEEP LEARNING PARAMETERS ..........................
            LOOK_BACK =  30 #731 (2 anys)     # quants dies amb anterioritat mires abans de predir
            FORECAST_RANGE = 7                # quants dies em de predir
            n_features = len(cc_predictors)   # quantes variables (sistema molt sensible)

            # HYPERPARAMETERS ............................................
            epochs = 10
            batch_size = 128
            validation = 0.1
            patience = 10

            X_train, y_train = split_sequence(scaled_train, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
            X_test, y_test = split_sequence(scaled_test, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)

            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)

            # MODEL TRAINING --------------------------------------------------------
            # GRU model .........................................................................................................
            print('\n >>>>> model 1: GRU')
            model_gru = create_model_gru(X_train)
            history = fit_model(model_gru,X_train, y_train, epochs, batch_size, validation, patience)
            yhat_gru = prediction(model_gru,X_test)
            y_test_inverse, yhat_gru_inverse = inverse_transform(y_test, yhat_gru, scaler)
            plt_model(y_test_inverse, yhat_gru_inverse,"GRU", objective = objective)

            # LSTM model ........................................................................................................
            print('\n >>>>> model 2: LSTM')
            model_lstm = create_model_lstm(X_train)
            history = fit_model(model_lstm,X_train, y_train, epochs, batch_size, validation, patience)
            yhat_lstm = prediction(model_lstm,X_test)
            y_test_inverse, yhat_lstm_inverse = inverse_transform(y_test, yhat_lstm, scaler)
            plt_model(y_test_inverse, yhat_lstm_inverse,"LSTM", objective = objective)

            # Bi-directional model .............................................................................................
            print('\n >>>>> model 3: Bi-directional')
            model_bilstm = create_model_bilstm(X_train)
            history = fit_model(model_bilstm,X_train, y_train, epochs, batch_size, validation, patience)
            yhat_bilstm = prediction(model_bilstm,X_test)
            y_test_inverse, yhat_bilstm_inverse = inverse_transform(y_test, yhat_bilstm, scaler)
            plt_model(y_test_inverse, yhat_bilstm_inverse,"Bi-directional_LSTM", objective = objective)

            # Encoder-decoder LSTM model .......................................................................................
            print('\n >>>>> model 4: Encoder-decoder LSTM')
            model_enc_dec = create_model_enc_dec(X_train)
            history = fit_model(model_enc_dec,X_train, y_train, epochs, batch_size, validation, patience)
            yhat_endelstm = prediction(model_enc_dec,X_test)
            y_test_inverse, yhat_endelstm_inverse = inverse_transform(y_test, yhat_endelstm, scaler)
            plt_model(y_test_inverse, yhat_endelstm_inverse,"ENCODER_DECODER_LSTM", objective = objective)

            # CNN-LSTM Encoder-Decoder model ...................................................................................
            print('\n >>>>> model 5: CNN-LSTM Encoder-Decoder')
            model_enc_dec_cnn = create_model_enc_dec_cnn(X_train)
            history = fit_model(model_enc_dec_cnn, X_train, y_train, epochs, batch_size, validation, patience)
            yhat_cnnlstmende = prediction(model_enc_dec_cnn,X_test)
            y_test_inverse, yhat_cnnlstmende_inverse = inverse_transform(y_test, yhat_cnnlstmende, scaler)
            plt_model(y_test_inverse, yhat_cnnlstmende_inverse,"Encoder_DECODER_CNN_LSTM", objective = objective)

            # Vector-Output model .............................................................................................
            print('\n >>>>> model 6: Vector-Output')
            model_vector_output = create_model_vector_output(X_train)
            history = fit_model(model_vector_output, X_train, y_train, epochs, batch_size, validation, patience)
            yhat_veout = prediction(model_vector_output,X_test)
            y_test_inverse, yhat_veout_inverse = inverse_transform(y_test, yhat_veout, scaler)
            plt_model(y_test_inverse, yhat_veout_inverse,"Vector_Output", objective = objective)

            # Multi-Head CNN-LSTM model ......................................................................................
            print('\n >>>>> model 7: Multi-Head CNN-LSTM')
            multi_head_cnn_lstm_model = create_model_multi_head_cnn_lstm(X_train)
            history = fit_model(multi_head_cnn_lstm_model, X_train, y_train, epochs, batch_size, validation, patience)
            yhat_muhecnnlstm = prediction(multi_head_cnn_lstm_model,X_test)
            y_test_inverse, yhat_muhecnnlstm_inverse = inverse_transform(y_test, yhat_muhecnnlstm, scaler)
            plt_model(y_test_inverse, yhat_muhecnnlstm_inverse, "Multi-Head CNN-LSTM", objective = objective)  

            # MODEL EVALUATION --------------------------------------------------------
            print('\n >>>>> EVALUATION OF THE DIFFERENT MODELS')

            # GRU model .........................................................................................................
            print('\n >>>>> model 1: GRU')
            evaluate_forecast(y_test_inverse, yhat_gru_inverse)

            # LSTM model ........................................................................................................
            print('\n >>>>> model 2: LSTM')
            evaluate_forecast(y_test_inverse, yhat_lstm_inverse)

            # Bi-directional model .............................................................................................
            print('\n >>>>> model 3: Bi-directional')
            evaluate_forecast(y_test_inverse, yhat_bilstm_inverse)

            # Encoder-decoder LSTM model .......................................................................................
            print('\n >>>>> model 4: Encoder-decoder LSTM')
            evaluate_forecast(y_test_inverse, yhat_endelstm_inverse)

            # CNN-LSTM Encoder-Decoder model ...................................................................................
            print('\n >>>>> model 5: CNN-LSTM Encoder-Decoder')
            evaluate_forecast(y_test_inverse, yhat_cnnlstmende_inverse)

            # Vector-Output model .............................................................................................
            print('\n >>>>> model 6: Vector-Output')
            evaluate_forecast(y_test_inverse, yhat_veout_inverse)

            # Multi-Head CNN-LSTM model ......................................................................................
            print('\n >>>>> model 7: Multi-Head CNN-LSTM')
            evaluate_forecast(y_test_inverse, yhat_muhecnnlstm_inverse)

            # PLOT ----------------------------------------------------------------------------------------------------------
            # plot models
            plt.figure(figsize=(20, 10))
            plt.plot(pd.DataFrame(y_test_inverse)[[col_idx]], label='True Values', linewidth=2.5)
            plt.plot(pd.DataFrame(yhat_gru_inverse)[[col_idx]], label='Pred. GRU', linestyle='--')
            plt.plot(pd.DataFrame(yhat_lstm_inverse)[[col_idx]], label='Pred. LSTM', linestyle='--')
            plt.plot(pd.DataFrame(yhat_bilstm_inverse)[[col_idx]], label='Pred. BiLSTM', linestyle='--')
            plt.plot(pd.DataFrame(yhat_endelstm_inverse)[[col_idx]], label='Pred. Enc-Dec LSTM', linestyle='--')
            plt.plot(pd.DataFrame(yhat_cnnlstmende_inverse)[[col_idx]], label='Pred. CNN-LSTM Enc-Dec', linestyle='--')
            plt.plot(pd.DataFrame(yhat_veout_inverse)[[col_idx]], label='Pred. Vector Output', linestyle='--')
            plt.plot(pd.DataFrame(yhat_muhecnnlstm_inverse)[[col_idx]], label='Pred. Mult-Head CNN-LSTM', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Real vs. Predicted Values MODELS')
            plt.legend()
            plt.show()

            print("Deep learning models training completed successfully!")'''
