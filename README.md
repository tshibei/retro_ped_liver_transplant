# retro_ped_liver_transplant

## Objective
Fit data with CURATE.AI models, compute predictions and prediction errors, compute dose recommendations.
Plot and analyse results.

## Getting Started
Install dependencies required in `requirements.txt`

## Usage
### Implement CURATE.AI models
```
python implement_CURATE.py
```
Add `--cross_val_method CV` to implement pop tau models with CV method instead of LOOCV (default).  
Add `--dose evening` to implement CURATE.AI models with evening dose instead of effective 24-hour dose.  
  
  The outputs are:
- *CURATE_results.xlsx* containing results of CURATE.AI models without pop tau,  
- *CURATE_results (of pop tau models only using <cross_val_method>).xlsx* containing results of CURATE.AI models with pop tau,  
- *experiments_to_find_pop_tau (by <cross_val_method>).xlsx* containing results from <cross_val_method> to find pop tau,  
- *all_data_total.xlsx* containing all patient data labeled as ideal or non-ideal for analysis, and  
- *dose_recommendations.xlsx* containing CURATE.AI dose recommendations.  

### Plot and analyse results
To plot figures. Valid figure numbers are `2`, `5`, `6`, or `7`.
```
python plotting.py --figure fig_<figure number>
```
  
  To annalyse results. Valid analysis names are `patient_population`, `technical_perf_metrics`, `clinically_relevant_perf_metrics`, `effect_of_CURATE`, or `fig_4_values`.
```
python plotting.py --analysis <analysis name>
```
