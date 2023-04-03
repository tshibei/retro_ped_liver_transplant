# retro_ped_liver_transplant

## Objective
Fit data to the CURATE.AI model, compute predictions and prediction errors, compute dose recommendations.
Plot and analyse results.

## Getting Started
Install dependencies required in `requirements.txt`

## Usage

### Implement CURATE.AI models
```
python implement_CURATE.py
```  
  The outputs are:
- *CURATE_results.xlsx* containing results of the CURATE.AI model, 
- *all_data.xlsx* containing all patient data labeled as ideal or non-ideal for analysis, and  
- *dose_recommendations.xlsx* containing CURATE.AI dose recommendations.  

### Plot and analyse results
To plot figures. Valid figure numbers are `2`, `5`, `6`, or `7`.
```
python plotting.py --figure fig_<figure number>
```
  
  To analyse results. Valid analysis names are `patient_population`, `technical_perf_metrics`, `clinically_relevant_perf_metrics`, `effect_of_CURATE`, or `fig_4_values`.
```
python plotting.py --analysis <analysis name>
```
