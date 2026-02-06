# Data quality study


## Column analysis

Load the datasets:
  hplt-3/fin_Latn                                                                                                                                                                      
  hplt-3/deu_Latn                                                                                                                                                                      
  nemotron-cc/high_actual                                                                                                                                                              
  finepdfs/fin_Latn
  finepdfs/deu_Latn
  fineweb-2/deu_Latn
  fineweb-2/fin_Latn

from https://huggingface.co/datasets/openeurollm/propella-annotations using `load_dataset` from `datasets` HF library.

For each dataset, I want you to compute histogram on all columns except for `one_sentence_description` column which 
should not be loaded to not fill memory. Load data by batch as done in analyse_data.py

Compute the histograms into data/analysis/{dataset}/histogram.csv

After computing is done, load this data and generate plot:
* showing a stackplot of the proportion of content_quality, ylabel should be "Proportion (%)", xlabel should be dataset
* showing a barplot of the number of document per content_quality, ylabel should be "Number of documents", xlabel should
be dataset, hue should be quality, title should be "Document quality count per dataset."
* also generate similar plots for information_density, audience_level, commercial_bias, content_safety, technical_content, regional_relevance, country_relevance
* generate a plot with 2 columns and one row per columns


- [ ] check quality histogram on nemotron-cc, hplt-3 and fw-edu-2
  - [ ] all datasets loaded with `from datasets import load_dataset`
  - [ ] compute histogram on columns values like in analyse_data.py and store values in csv
  - [ ] plot of histogram for percentage and proportion


## Classifier

Goal: check performance of classifier to predict propella `content_quality` column given others 
Reason:
- could allow to get quality distribution and allow to do importance sampling
- could allow to understand what the quality column "means"

TODOs:
- [x] start by reading annotate_data.py to understand how to fit an autogluon model
- [x] move function to generate features and fit autogluon model to an utils.py
- [x] create predict_propella_quality.py which loads nemotron-cc-sample and shows the accuracy when predicting
`content_quality` given other columns

Results:
- **Accuracy: 80%** predicting content_quality from other propella features
- Most important feature: `information_density` (by far the most predictive)
- Other important features: `content_integrity`, `audience_level`, `content_length`, `educational_value`, `reasoning_indicators`
- Confusion matrix shows most errors are between adjacent quality levels (adequate↔good)
- Model rarely makes extreme errors (e.g., excellent↔unacceptable)

Files created:
- `utils.py` - shared functions for feature preparation and model training
- `predict_propella_quality.py` - classifier script
- `figures/propella_quality_confusion_matrix.pdf` - confusion matrix visualization
- `figures/propella_quality_feature_importance.pdf` - feature importance plot

Notes: