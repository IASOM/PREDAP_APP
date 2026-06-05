""" 
In this file we define the class used to evaluate the models already trained and obtain valuable metrics 
like MAE, RMSE, MSE
"""
from datetime import datetime

from config.base_transformer_config import BaseTransformerConfig
from utils import data_preparation
from dataclasses import dataclass, field
import numpy as np
import os


from residual_multivariate_transformers.utils_residual_transformer import load_base_model_transformer, load_trained_model

from main_train_univ_transformer_class import TransformerUnivConfig, UnivariateTransformerPipeline
from main_train_diagnostic_residual_transformer_class import DiagnosticResidualTransformerConfig, DiagnosticResidualTransformerPipeline
from main_train_seasonal_residual_transformer_class import SeasonalResidualTransformerConfig, SeasonalResidualTransformerPipeline

@dataclass
class ConfigEvalTransformers(BaseTransformerConfig):
    """ Configuration class for the eval of the models """



class EvalTransformers:

    def __init__(self,config:ConfigEvalTransformers):
        self.config = config

        self.univ_model = None
        self.diagnostics_model = None
        self.seasonal_model = None 



    def compute_predictions(self, UnivariateClass: UnivariateTransformerPipeline, DiagnosticsClass:DiagnosticResidualTransformerPipeline,SeasonalClass: SeasonalResidualTransformerPipeline , forecast:int, lookback:int):

        X_univ, Y_univ = UnivariateClass.prepare_data(train=True)
        X_univ_test, Y_univ_test = UnivariateClass.prepare_data(train=False)
        univ_model_name = UnivariateClass.config.get_model_name()

        
        model_folder = self.config.model_folder
        predictions_train_univ, predictions_test_univ = load_base_model_transformer(X_univ, X_univ_test, base_path = model_folder, base_model_name = univ_model_name)
        
        model_residuals = Y_univ - predictions_train_univ
        return predictions_train_univ, predictions_test_univ, model_residuals
    
    def compute_diagnostic_predictions(self, DiagnosticsClass:DiagnosticResidualTransformerPipeline,predictions_train_univ:np.ndarray, predictions_test_univ:np.ndarray, residuals:np.ndarray, forecast:int, lookback:int):
       
        X_train_diagnostic, X_test_diagnostic = DiagnosticsClass.prepare_covariate_data()
        diagnostic_model_name = DiagnosticsClass.config.get_diagnostic_residual_model_name()

        model_folder = self.config.model_folder
        
        predictions_residuals_train_diagnostics, predictions_residuals_test_diagnostics = load_base_model_transformer(X_train = X_train_diagnostic,  X_test = X_test_diagnostic, base_path = model_folder, base_model_name = diagnostic_model_name)
        
        predictions_residuals_test_diagnostics = np.squeeze(predictions_residuals_test_diagnostics, axis=-1)
        predictions_residuals_train_diagnostics = np.squeeze(predictions_residuals_train_diagnostics, axis =-1)

        predictions_train_diagnostics = predictions_train_univ + predictions_residuals_train_diagnostics
        predictions_test_diagnostics = predictions_test_univ + predictions_residuals_test_diagnostics

        return predictions_test_diagnostics,predictions_train_diagnostics, predictions_residuals_test_diagnostics

    def compute_seasonal_predictions(self, SeasonalClass:SeasonalResidualTransformerPipeline,predictions_train_diagnostics:np.ndarray,predictions_test_diagnostics:np.ndarray, residuals:np.ndarray, forecast:int, lookback:int):
        SeasonalClass.predictions_train = predictions_train_diagnostics
        SeasonalClass.predictions_test = predictions_test_diagnostics
        X_train_seasonal, X_test_seasonal = SeasonalClass.prepare_covariate_data()
        seasonal_model_name = SeasonalClass.config.get_seasonal_residual_model_name()

        model_folder = self.config.model_folder
       
        predictions_residuals_train_seasonal, predictions_residuals_test_seasonal = load_base_model_transformer(X_train = X_train_seasonal,  X_test = X_test_seasonal, base_path = model_folder, base_model_name = seasonal_model_name)
        predictions_residuals_test_seasonal = np.squeeze(predictions_residuals_test_seasonal, axis=-1)
        predictions_test_seasonal = predictions_test_diagnostics + predictions_residuals_test_seasonal
        return predictions_test_seasonal, predictions_residuals_test_seasonal



    def setup_classes(self,code: str, forecast:int, lookback:int, ff_dim:int, learning_rate:float, univ_model_name:str, diagnostic_model_name:str, seasonal_model_name:str):
        univ_config = TransformerUnivConfig(  
                                    lookback=lookback,
                                    forecast=forecast,
                                    code=code,
                                    ff_dim = ff_dim, 
                                    learning_rate = learning_rate,
                                   
                                    )
        

        UnivClass = UnivariateTransformerPipeline(config=univ_config)

        diagnostics_config = DiagnosticResidualTransformerConfig(
                                    lookback=lookback,
                                    forecast=forecast,
                                    code=code,
                                    ff_dim = ff_dim, 
                                    learning_rate = learning_rate,
                                    
                                )
        DiagnosticsClass = DiagnosticResidualTransformerPipeline(
                                    config = diagnostics_config                        
                                )

        seasonal_config = SeasonalResidualTransformerConfig(
                                    lookback=lookback,
                                    forecast=forecast,
                                    code=code,
                                    ff_dim = ff_dim, 
                                    learning_rate = learning_rate,
                                    
                                )

        SeasonalClass = SeasonalResidualTransformerPipeline(config = seasonal_config)


        return UnivClass, DiagnosticsClass, SeasonalClass

    def run_eval_x_step(self, lookback_list:list[int], forecast_list:list[int], code:str, ff_dim:int, learning_rate:float):
        results_list = []
        found_lookbacks = []
        for lookback in lookback_list:
            for forecast in forecast_list:
                univ_model_name = (f'{code}_base_transformer_'
                                   f'{forecast}fh_{ff_dim}ff_{lookback}lb_'
                                   f'{learning_rate}lr.keras')

                diagnostic_model_name = (f'{code}_DIAGNOSTIC_RESIDUALS_LEARNING_'
                                         f'{forecast}fh_{ff_dim}ff_{lookback}lb_'
                                         f'{learning_rate}initlr.keras')

                seasonal_model_name = (f'{code}_SEASONAL_RESIDUALS_LEARNING_'
                                       f'{forecast}fh_{ff_dim}ff_{lookback}lb_'
                                       f'{learning_rate}initlr.keras')

                
                if univ_model_name in os.listdir(self.config.model_folder):

                    UnivClass, DiagnosticsClass, SeasonalClass = self.setup_classes(code, forecast, lookback, ff_dim, learning_rate, univ_model_name, diagnostic_model_name, seasonal_model_name)

                    predictions_train_univ, predictions_test_univ, residuals_univ = self.compute_predictions(UnivClass, DiagnosticsClass, SeasonalClass, forecast, lookback)
                    predictions_test_diagnostics,predictions_train_diagnostics, residuals_test_diagnostics = self.compute_diagnostic_predictions(DiagnosticsClass,predictions_train_univ, predictions_test_univ,residuals_univ, forecast, lookback)
                    predictions_test_seasonal, residuals_seasonal = self.compute_seasonal_predictions(SeasonalClass, predictions_train_diagnostics, predictions_test_diagnostics,residuals_test_diagnostics, forecast, lookback)



                    results_list.append(predictions_test_seasonal)
                    found_lookbacks.append(lookback)
                else:
                    print(f"model:{univ_model_name} not found")

        return results_list, found_lookbacks

    def compare_eval_steps(self,results_list:list[np.ndarray],found_lookbacks:list[int], max_shape:int):
        
        results_max = results_list[-1]
        max_lookback = found_lookbacks[-1]
        assert results_max.shape[1] == max_shape 

        UnivClass, DiagnosticsClass, SeasonalClass = self.setup_classes(code = self.config.code, forecast = max_shape, lookback = max_lookback, ff_dim = self.config.ff_dim, learning_rate = self.config.learning_rate, univ_model_name = None, diagnostic_model_name = None, seasonal_model_name = None)

        X_univ, Y_univ = UnivClass.prepare_data(train = False)
        X_univ_train_orig, Y_univ_train_orig = UnivClass.prepare_data_not_normalized(train = True)
        X_univ_test_orig, Y_univ_test_orig = UnivClass.prepare_data_not_normalized(train = False)

       
        X_univ_train_orig = np.squeeze(X_univ_train_orig, axis= 2)
        





        # ✅ NEW: Initialize tracking variables
        comparison_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "code": self.config.code,
                "max_forecast_horizon": max_shape,
            },
            "detailed_results": [],
            "summary": {
                "times_mae_res_lower": 0,
                "total_comparisons": len(results_list),
                "percentage_improvements": 0.0,
                "average_improvement_magnitude": 0.0,
                "improvements_list": [],
                "best_forecast_horizon": None,
                "best_mae_improvement": 0.0
            }
        }
        
        improvements_list = []
        times_min_is_lower = 0
        best_improvement = 0.0
        best_horizon = None
        
        print("\n" + "="*70)
        print("📊 COMPARISON EVALUATION RESULTS")
        print("="*70)
        for results, lookback in zip(results_list, found_lookbacks):


            results_len = results.shape[1]
            res_max = results_max[:, :results_len]
            #res_max = results_max[:Y_univ_test_orig.shape[0],:results_len]
            start_cut = max_lookback - lookback
            end_cut = -(max_shape - results_len) if (max_shape - results_len) !=0 else None
            res = results[start_cut:end_cut, :results_len]
            #results = results[:Y_univ_test_orig.shape[0],:results_len]
            
            
            Y = Y_univ_test_orig[:,:results_len]
            X_orig = X_univ_train_orig[:,:results_len]

            new_scaler = UnivClass.config.scaler
            new_scaler.fit(X_orig)
            res_max_orig = new_scaler.inverse_transform(res_max)
            results_orig = new_scaler.inverse_transform(res)

            mae_res = np.mean(np.abs(Y - results_orig))
            mse_res = np.mean((Y - results_orig)**2)
            rmse_res = np.sqrt(mse_res)


            mae_max_res = np.mean(np.abs(Y - res_max_orig))
            mse_max_res = np.mean((Y - res_max_orig)**2)
            rmse_max_res = np.sqrt(mse_max_res)

            print(f"Results for forecast horizon {results_len}:")
            print(f"MAE: {mae_res:.4f}, MSE: {mse_res:.4f}, RMSE: {rmse_res:.4f}")
            print(f"Max Model Results for forecast horizon {max_shape}:")
            print(f"MAE: {mae_max_res:.4f}, MSE: {mse_max_res:.4f}, RMSE: {rmse_max_res:.4f}")

            # ✅ NEW: Calculate improvement metrics
            min_is_better = mae_res < mae_max_res
            improvement_magnitude = mae_max_res - mae_res if min_is_better else 0.0
            improvement_percentage = (improvement_magnitude / mae_max_res * 100) if mae_max_res > 0 else 0.0
            
            if min_is_better:
                times_min_is_lower += 1
                improvements_list.append(improvement_magnitude)
                
                # Track best improvement
                if improvement_magnitude > best_improvement:
                    best_improvement = improvement_magnitude
                    best_horizon = results_len
            
            # ✅ NEW: Store detailed results
            detailed_result = {
                "forecast_horizon": results_len,
                "lookback": lookback,
                "mae_res": float(mae_res),
                "mse_res": float(mse_res),
                "rmse_res": float(rmse_res),
                "mae_max_res": float(mae_max_res),
                "mse_max_res": float(mse_max_res),
                "rmse_max_res": float(rmse_max_res),
                "is_mae_lower": bool(min_is_better),
                "improvement_magnitude": float(improvement_magnitude),
                "improvement_percentage": float(improvement_percentage),
                "relative_performance": "MIN is BETTER" if min_is_better else "MIN is WORSE"
            }
            
            comparison_results["detailed_results"].append(detailed_result)
            
            # ✅ ENHANCED: Print results with improvement info
            print(f"\n📈 Forecast Horizon {results_len}:")
            print(f"\n Lookback {lookback}")
            print(f"   MAE: {mae_res:.6f} | MSE: {mse_res:.6f} | RMSE: {rmse_res:.6f}")
            print(f"📊 Max Model (truncated to {results_len}):")
            print(f"   MAE: {mae_max_res:.6f} | MSE: {mse_max_res:.6f} | RMSE: {rmse_max_res:.6f}")
            
            if min_is_better:
                print(f"✅ IMPROVEMENT: MIN-MAE is {improvement_magnitude:.6f} lower ({improvement_percentage:.2f}% better)")
            else:
                print(f"❌ WORSE: MIN-MAE is {abs(improvement_magnitude):.6f} higher")
            print("-" * 50)

        # ✅ NEW: Calculate summary statistics
        average_improvement = np.mean(improvements_list) if improvements_list else 0.0
        percentage_improvements = (times_min_is_lower / len(results_list)) * 100
        
        # Update summary
        comparison_results["summary"].update({
            "times_mae_res_lower": times_min_is_lower,
            "percentage_improvements": percentage_improvements,
            "average_improvement_magnitude": float(average_improvement),
            "improvements_list": [float(x) for x in improvements_list],
            "best_forecast_horizon": best_horizon,
            "best_mae_improvement": float(best_improvement)
        })
        
        # ✅ NEW: Print summary
        print("\n" + "="*70)
        print("📊 SUMMARY STATISTICS")
        print("="*70)
        print(f"🎯 Times MIN-MAE was lower: {times_min_is_lower} out of {len(results_list)} comparisons")
        print(f"📈 Percentage of improvements: {percentage_improvements:.1f}%")
        print(f"📊 Average improvement magnitude: {average_improvement:.6f}")
        
        if best_horizon:
            print(f"🏆 Best performing forecast horizon: {best_horizon} (improvement: {best_improvement:.6f})")
        
        if improvements_list:
            print(f"📈 Improvement range: {min(improvements_list):.6f} to {max(improvements_list):.6f}")
            print(f"📊 Median improvement: {np.median(improvements_list):.6f}")
        else:
            print("❌ No improvements found")


    def run_main_eval(self):
        results_list, found_lookbacks = self.run_eval_x_step([7,14,30,60,182,365], [7,14,30,60,182,365], self.config.code, self.config.ff_dim, self.config.learning_rate)
        self.compare_eval_steps(results_list, found_lookbacks, max_shape=365)
        #[7,14,30,60,182,365]



            




if __name__ == "__main__":
    
    config = ConfigEvalTransformers()

    EvalClass = EvalTransformers(ConfigEvalTransformers)

    EvalClass.run_main_eval()




    






       