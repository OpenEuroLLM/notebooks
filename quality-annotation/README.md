# quality annotation analysis

Measure how much propella annotation are predictive of nemotron-cc quality annotations.

Eg given a dataframe like this, we use all columns but `warc_record_id` to predict `quality`.
```
                             warc_record_id content_integrity     content_ratio content_length                                                                                  one_sentence_description                 content_type                    business_sector  technical_content information_density content_quality audience_level commercial_bias time_sensitivity content_safety educational_value reasoning_indicators  pii_presence    regional_relevance country_relevance  quality
0  5e730137-c560-4aeb-92e1-8d98aec5c34e          fragment  complete_content        minimal                                                                A list of timestamps with dates and times.            [structured_data]                            [other]    [non_technical]               dense            poor        general            none        evergreen           safe              none                 none        no_pii       [indeterminate]            [none]        2
1  629cb7e8-9d8b-4aaf-8fec-6cf7d7a8ab2c          complete  complete_content          brief                                   Promotional description of adult video actress Alexis Grace from Miami.              [transactional]              [media_entertainment]    [non_technical]                thin            poor        general  pure_marketing  slowly_changing           nsfw              none                 none        no_pii      [north_american]   [united_states]        1
2  3d38b8f3-dfe4-4522-b6b4-9142b70093e6          complete    mostly_content        minimal      A logical reasoning question asks for the underlying assumption in an argument about job applicants.  [qa_structured, analytical]                 [education_sector]    [non_technical]               dense            good        general            none        evergreen           safe          moderate           analytical        no_pii  [culturally_neutral]            [none]        2
3  0e19ab0e-9973-40ff-a33e-a6b3621f445a          complete  complete_content       moderate  Personal blog post answering a challenge to describe the author's personality traits using the alphabet.                   [creative]                 [general_interest]    [non_technical]            moderate        adequate        general            none        evergreen           safe              none                 none  contains_pii      [north_american]   [united_states]        1
4  6f98c1b4-a896-4da1-a395-921b02dc8812          complete  complete_content        minimal                             Product description for a strawberry, pineapple, and lemon flavored e-liquid.              [transactional]  [retail_commerce, consumer_goods]  [basic_technical]               dense            good        general  pure_marketing  slowly_changing           safe              none                 none        no_pii  [culturally_neutral]            [none]        0
```

## Installation

Requires Python 3.12+. This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd quality-annotation

# Install dependencies
uv sync
```

## Usage

```bash
uv run python annotate_data.py
```

which will prints among others the accuracy/MAE of the classifer

```
Accuracy: 0.4778
MAE: 0.7636
```
(note that accuracy and MAE of a constant mean predictor is 0.2 and 1.2 respectively).

in addition to a feature importance report:
```
                                      importance    stddev  ...  p99_high   p99_low
educational_value                        0.10576  0.006317  ...  0.118767  0.092753
business_sector_education_sector         0.02752  0.005973  ...  0.039818  0.015222
time_sensitivity                         0.02316  0.003769  ...  0.030921  0.015399
content_length                           0.01488  0.001622  ...  0.018220  0.011540
commercial_bias                          0.01448  0.002697  ...  0.020032  0.008928
```

