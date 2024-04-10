# Abstract

The model is currently under development.
The current dataset contains 49 million LinkedIn profiles.
The model predicts the probability of a profile leaving the company within the next four years or staying.
The problem is tackled as a multi-label problem with an output vector yielding probabilities for each of the five classes.
The model consists of two stages: a classifier assuming independence amongst the classes and a secondary stage exploiting logical constraints to model dependencies amongst the classes.

# TOC

<!--toc:start-->
- [Abstract](#abstract)
- [TOC](#toc)
- [Current project status](#current-project-status)
- [Project goal](#project-goal)
- [Problem Description](#problem-description)
- [Ethical considerations](#ethical-considerations)
- [Literature Context](#literature-context)
- [Data](#data)
  - [Label Encoding](#label-encoding)
    - [Data labeling](#data-labeling)
- [Key modeling challenges](#key-modeling-challenges)
  - [Imbalanced labels](#imbalanced-labels)
  - [Label dependency](#label-dependency)
  - [Multi modal data](#multi-modal-data)
- [Data preprocessing](#data-preprocessing)
  - [Text preprocessing](#text-preprocessing)
- [Machine learning model](#machine-learning-model)
  - [Stage 1: Prediction assuming label independence](#stage-1-prediction-assuming-label-independence)
  - [Stage 2: Modeling dependencies using constraints](#stage-2-modeling-dependencies-using-constraints)
    - [Constraint notation](#constraint-notation)
    - [Dependency modeling](#dependency-modeling)
  - [Implementation notes](#implementation-notes)
- [Model training](#model-training)
  - [Ablation study](#ablation-study)
- [A data science perspective](#a-data-science-perspective)
  - [Feature engineering](#feature-engineering)
  - [Model adaptations](#model-adaptations)
<!--toc:end-->

# Current project status

Currently, ablation studies are being performed. The project will be updated after both ablation studies conclude.

# Project goal

The goal of the project is to implement and test the limits of novel machine learning algorithms in a complex multi-label real-world problem. 
Therefore, the data treatment and feature engineering are kept to a minimum. The project will suffer from excess complexity and suboptimally prepared data.
The steps required for a better data science-driven solution can be found under the [data science perspective](#a-data-science-perspective).


# Problem Description

The presented model attempts to solve the following hypothetical business case:

The online recruitment market was valued in 2023 at 29660 million USD, with LinkedIn being one of the largest players [Global Online Recruitment Market Research Report 2024](https://www.marketgrowthreports.com/global-online-recruitment-market-26511727).
Currently, the recruitment process follows a targeted mass mail approach which causes most offers for top talent to look like spam.
By predicting which current top talent is most likely to leave their current employer within the next year, the next two years, the next three years or the next four years, the presented platform allows more efficient targeting
thus directly reducing the per unit cost basis.
Additionally, hiring managers can use the platform to identify promising targets to build a personal relationship with prior to the job offer.
Thereby significantly increasing their conversion rate and ultimately RoI by better targeting and more efficient deployment of available hiring managers.

As a secondary use case, companies can use the model to screen potential applicants or current employees for company loyalty to better predict their hiring demand for the next period.

The presented solution should take a LinkedIn profile as input and output a probability estimate of a candidate leaving the company in
year one, year two, year three, and year four and not leaving the company within four years. It should support the most common languages.


# Ethical considerations

In an effort to prevent harms outlined by [Thomas, D. R., Pastrana, S., Hutchings, A., Clayton, R., & Beresford, A. R. (2017, November)](https://dl.acm.org/doi/abs/10.1145/3131365.3131389)
the project will adopt the following policies:

- No GitHub history or dvc history will be kept to prevent accidental data leakage.
- No collected data will be made publicly available (interested stakeholders with a justifiable interest can approach me for access to relevant parts of the data).
- The dataset is treated with the [AI privacy toolkit](https://doi.org/10.1016/j.softx.2023.101352).
- All potential data supplied for tests will be synthetically generated.
- No individual statistics pointing to specific groups will be published.
- To prevent ill-gotten financial gain, the model will be artificially kept in the year 2021, thus rendering it inapt for predictions in the current year.
- The project will be released under a restrictive licence preventing 3rd parties from using the model for commercial purposes.

The author acknowledges that the concerns (proof of maximising benefits to society, informed consent, equitable subject selection,
transparent and accountable research) of [Boustead, A. E., & Herr, T. (2020)](https://doi.org/10.1017/S1049096520000323) are not elevated
by these measures. However, the author is neither publishing the findings as peer-reviewed research nor does he believe the prescriptions by 
Bousted & Herr are validly argued, let alone soundly justified.

# Literature Context

None of the surveyed articles uses unstructured text as part of their feature set. All surveyed studies encountered imbalanced datasets.
Furthermore, the largest dataset presented in the literature by [de Jesus, A. C. C., Júnior, M. E. G., & Brandão, W. C. (2018, April)](https://doi.org/10.1145/3167132.3167320) is
around 120.000 profiles before data cleaning. No author framed the problem as a multi-label problem. Relevant features and
relevant literature is presented in [Literature context of potential features](./documentation/EDA.md#literature-context-of-potential-features)

# Data

Definition of a valid profile: A profile that has a current employer and started their current job between 1990 and 2021.
The initial dataset contains a total of 450 million profiles collected from publicly available sources.
After data cleaning and removal of the non-valid profiles, the dataset shrank to 91 million profiles.
Each profile consists of 153 features, including a mix of categorical values, numerical data, and unstructured text.
A more granular breakdown of the data, data treatment and a full data schema can be found [here](./documentation/data.md)

## Label Encoding

Definition: A profile has changed jobs during the last n years if during the last n years the profile employer changed
such that if there had not been a new employer the profile would be considered unemployed.

### Data labeling

Profiles are labelled as one of the following two mutually exclusive groups. Either a profile stayed with the company or left the company during the last four years.
Leaving the company is cumulative, such that leaving in year `i` implies the profile would also have been left in year `i+1`.

Thus, all elements of the set `leaving in year i` are a subset of `leaving in year i+1`.
Therefore, there will be non-uniformly distributed labels, which need to be confirmed in the EDA and subsequently addressed during modeling.

A profile is considered to have `stayed with the company` if it did not leave the company during the last four years.

A profile is considered to have `left the company in year i` if during the period `2021 - (2021 -i)`
the current employer changed where the period excludes the current year except for year 1.

<details>
<summary>Example python labeling code</summary>

```python
def create_labels(*,profile_df: pd.DataFrame,
                  job_start_prefix: str ='7_job_start',
                  company_name_prefix: str ='7_company_name_',) -> pd.DataFrame:
    job_starts = [word for word in profile_df.columns if 
                  word.startswith(job_start_prefix)]
    company_names = [word for word in profile_df.columns if
                    word.startswith(company_name_prefix)]
    profile_df.reset_index(inplace=True)
    profile_df[job_starts] = profile_df[job_starts].fillna(value=0)
    data = pd.DataFrame(index = profile_df.index,
                        columns = ['years_until_first_company_change',
                                   'years_until_second_company_change',])
    data['years_until_first_company_change'] = 2021 - profile_df[['0_job_start_date']]
    data['years_until_second_company_change'] = np.sort(profile_df[job_starts],
                                                        axis=1)[:, -2]
    labels = pd.DataFrame(index = profile_df.index,
                          columns=['9_stayed_with_company',
                                    '9_left_year_1',
                                    '9_left_year_2',
                                    '9_left_year_3',
                                    '9_left_year_4',])
    two_employment_changes = data['years_until_second_company_change'] != 0
    at_least_two_unique_employers = (profile_df[['0_job_company_name']+
                                               company_names].nunique(axis=1,
                                                                      dropna=True)
                                                                      > 1)
    labels['9_stayed_with_company'] = data['years_until_first_company_change'] > 4
    mask_stayed = data.loc[data['years_until_second_company_change'] == 0,].index
    labels.loc[mask_stayed,['9_stayed_with_company',]] = True
    labels.loc[two_employment_changes &
               at_least_two_unique_employers &
               (data['years_until_first_company_change'] <= 4) 
                ,['9_stayed_with_company', ]] = True
    inverted_stay_mask = (labels['9_stayed_with_company'] == True).values
    mask_left = labels.loc[inverted_stay_mask,].index
    labels.loc[mask_left, ['9_left_year_1',
                           '9_left_year_2',
                           '9_left_year_3',
                           '9_left_year_4']] = False
    labels.loc[two_employment_changes &
               at_least_two_unique_employers &
               (data['years_until_first_company_change'] == 4 ) 
                ,['9_left_year_4',]] = True
    labels.loc[two_employment_changes &
               at_least_two_unique_employers &
               (data['years_until_first_company_change'] == 3 )
                ,['9_left_year_3',
                  '9_left_year_4',]] = True
    labels.loc[two_employment_changes &
               at_least_two_unique_employers &
               (data['years_until_first_company_change'] == 2 ) 
                ,['9_left_year_2',
                  '9_left_year_3',
                  '9_left_year_4',]] = True
    labels.loc[two_employment_changes &
               at_least_two_unique_employers &
               (data['years_until_first_company_change'] <= 1 )
                ,['9_left_year_1',
                  '9_left_year_2',
                  '9_left_year_3',
                  '9_left_year_4',]] = True
    labels.fillna(value=False, inplace = True)
    return labels
```

</details>

Note: the example code only covers cases compliant with the outlined definition, edge cases such as leaving jobs while having multiple simultaneous employments
are not covered and will remain as nan values in the sample.

# Key modeling challenges

## Imbalanced labels

The labels are highly imbalanced, with the lowest occurring label making up only [2.2%](./documentation/EDA.md#overview-statistics).
Since the preferred solution of upsampling using [MLSOL](https://doi.org/10.48550/arXiv.2005.03240) or [MLSMOTE](10.1016/j.knosys.2015.07.019) to synthetically generate numerical and categorical features
and to substitute keywords in unstructured text failed due to the complex interdependence of features i.e. job summary, job title and job role, the dataset
had to be randomly downsampled. After downsampling, the total number of profiles dropped to 45 million, with the lowest occurring label making up roughly 4.4% of the total profiles.
Since the labels are overlapping, that's the best downsampling can do.

Imbalance metrics from [Charte, F., Rivera, A., del Jesus, M. J., & Herrera, F. (2013)](https://doi.org/10.1007/978-3-642-40846-5_16) after undersampling:

Imbalance ratio per label (IRLbl):

- year 1: 11.57
- year 2: 2.42
- year 3: 1.37
- year 4: 1.0
- stayed with company: 1.0

Mean imbalance ratio (MeanIR): 3.47

Coefficient of variation of IRLbl (CVIR): 5.99

Since a decision regarding resampling has already been made [SCUMBLE](https://doi.org/10.1007/978-3-319-07617-1_10) is omitted.

Furthermore, the multilabel stratified k-fold cross-validation proposed by [Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011)](https://doi.org/10.1007/978-3-642-23808-6_10)
will be used to ensure that every sample represents the minority labels.

## Label dependency

By construction the labels are highly dependent.

The `9_left_year_1` label predicts 4.4% of the observations from the labels `9_left_year_2`, `9_left_year_3` and `9_left_year_4`

The `9_left_year_2` label prediction 20.0% of the observations from the labels `9_left_year_3` and `9_left_year_4`

The `9_left_year_3` label predicts 36.1% of the observations from the label `9_left_year_4`

Therefore, it is paramount that the model considers label dependency.

## Multi modal data

After reviewing the [Modality heterogeneity matrix and Interaction heterogeneity matrix](https://doi.org/10.48550/arXiv.2203.01311)
late fusion is not considered due to the heterogeneity of modality signal strength across modalities.
Despite the promising results of [Gu, K., & Budhkar, A. (2021, June)](https://doi.org/10.18653/v1/2021.maiworkshop-1.10) data fusion will not be performed
with a Multimodal Adaptation Gate (MAG) proposed by [Rahman, W., Hasan, M. K., Lee, S., Zadeh, A., Mao, C., Morency, L. P., & Hoque, E. (2020, July)](https://doi.org/10.18653%2Fv1%2F2020.acl-main.214)
or with a weighted sum approach. Both approaches seem implausible apriori to caputre the relationships present
in the data. Instead, a basic concatenation of BERT embeddings (CLS token) and encoded features will be used.

# Data preprocessing

The numerical features (0_linkedin_connections, 0_year,...) are not normally distributed and, therefore, scaled using a min-max scalar.
Due to memory constraints, the entire dataset is split into 49 partial datasets, all containing 1 million profiles.
numerical features are scaled during runtime. The scaler is fitted on the training dataset.

Unstructured text columns are aggregated into text blocks by concatenation of text features with `[SEP]` tokens.

Ordinal, One-hot encodings and text tokenisation are also done at runtime.

## Text preprocessing

Unstructured text is preprocessed is kept due to limited computational resources to a minimum. Characters are converted to lowercase, all German special characters are expanded,
website links are removed, remaining special characters are removed and stopwords are removed. Additionally, text columns are trimmed in length.

# Machine learning model

The final model uses the CCN architecture proposed by [Giunchiglia, E., & Lukasiewicz, T. (2021)](https://doi.org/10.1613/jair.1.12850).
The model consists of two stages. In the first state, a multi-label classifier assumes independence amongst the labels.
In the second stage, dependencies amongst labels are modeled by using a set of hard logical constraints to mutate the prediction of the first-stage classifier.

## Stage 1: Prediction assuming label independence

The input dimension for the classifier is 7977 (concatenation of 10 bert cls embeddings + 1 numerical feature + 2 One-hot encodings + 1 ordinal encoding)
and the output dimension is 5.
The classifier is constructed as a series of linear layers, activation layers, norm layers and
dropout layers. In every hidden linear layer, the input dimension is constant.
The last layer outputs probability estimates for each class.
Since independence is assumed Binary Cross Entropy is used as a criterion.

<details>
<summary>Classifier model summary used in ablation study</summary>

```

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MLP                                      [32, 5]                   --
├─Sequential: 1-1                        [32, 5]                   --
│    └─Linear: 2-1                       [32, 6000]                47,868,000
│    └─LeakyReLU: 2-2                    [32, 6000]                --
│    └─BatchNorm1d: 2-3                  [32, 6000]                12,000
│    └─Dropout: 2-4                      [32, 6000]                --
│    └─Linear: 2-5                       [32, 6000]                36,006,000
│    └─LeakyReLU: 2-6                    [32, 6000]                --
│    └─BatchNorm1d: 2-7                  [32, 6000]                12,000
│    └─Dropout: 2-8                      [32, 6000]                --
│    └─Linear: 2-9                       [32, 6000]                36,006,000
│    └─LeakyReLU: 2-10                   [32, 6000]                --
│    └─BatchNorm1d: 2-11                 [32, 6000]                12,000
│    └─Dropout: 2-12                     [32, 6000]                --
│    └─Linear: 2-13                      [32, 6000]                36,006,000
│    └─LeakyReLU: 2-14                   [32, 6000]                --
│    └─BatchNorm1d: 2-15                 [32, 6000]                12,000
│    └─Dropout: 2-16                     [32, 6000]                --
│    └─Linear: 2-17                      [32, 6000]                36,006,000
│    └─LeakyReLU: 2-18                   [32, 6000]                --
│    └─BatchNorm1d: 2-19                 [32, 6000]                12,000
│    └─Dropout: 2-20                     [32, 6000]                --
│    └─Linear: 2-21                      [32, 6000]                36,006,000
│    └─LeakyReLU: 2-22                   [32, 6000]                --
│    └─BatchNorm1d: 2-23                 [32, 6000]                12,000
│    └─Dropout: 2-24                     [32, 6000]                --
│    └─Linear: 2-25                      [32, 5]                   30,005
│    └─Sigmoid: 2-26                     [32, 5]                   --
==========================================================================================
Total params: 228,000,005
Trainable params: 228,000,005
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 7.30
==========================================================================================


```

</details>


## Stage 2: Modeling dependencies using constraints

### Constraint notation

Every line in [constraints.txt](./data/constraints/constraints.txt) represents a logical implication where the left side of `<-` is the implied value called consequent (as a label index),
`<-` reads as implies and the right side of `<-` consists of variables which imply the consequent called antecedents. Antecedents can be either positive or negative.
A negative antecedent is denoted by an `n` before the index. There can be only one consequent (which in this project will always be positive) and at maximum n-1 antecedent.

Example: Let y be a 1x5 vector representing labels

Then `2 <- 1 n3` translates to if the second label (zero indexed labels) label is true and the fourth label is false then the third label is true
which would be equivalent to $<x,1,1,0,y> \impliedby <x,1,?,0,y> \quad x,y \in \{0,1\}$

### Dependency modeling
Let y be the label vector `<left year 1, left year 2, left year 3, left year 4, stayed at company>`

There exist two possible constraint sets $\pi_0$ and $\pi_1$ with $\pi_0 \subset \pi_1$.

$\pi_0 = \set{\text{left year i+1} \impliedby \text{left year i} \mid \text{i} \in \set{1,2,3\}  } \ \cup \set{\text{stayed at company} \impliedby \neg \ \text{left in year i} \mid \ \text{i} \in \set{1,2,3\} }$  

$\pi_1 = \pi_0 \cup \theta_0 \cup \theta_1$

$\theta_0 = \set{\text{left year i+1} \impliedby \bigcup_{i \le i} \text{left year i} }$

$\theta_1 = \set{ \text{stayed at company} \impliedby \bigcup_{i} \neg \ \text{left year i} \mid \forall \ i \le 4}$

Which constraint set is appropriate will be determined after the ablation study.

## Implementation notes

It would've been preferable to use polars for the dataset class.
This is currently infeasible since the current implementation of K-fold stratification returns a set of indicies and polars does not support selection by index.

For use in a pipeline the preproceesor module should reimplement text preprocessing using polars to apply regex operations with [pyre2](https://github.com/andreasvc/pyre2).
Additionally, a separate  machine learning model should be trained to impute missing values in features that are directly fed into the CCN model.

The issue of possible inconsistency  in One-hot encodings (case a random fold is missing a category in validation or training)
is addressed firstly probabilistically by partitioning the total dataset into partial datasets each with a size of roughly
each containing 1 million profiles which given a validation set proportion of 30% would imply a probability of less
than 0.05% to encounter a random fold with at least one missing category.
Secondly, the issue is addressed deterministically by padding the input vector to the input size of the first linear layer.

The presented implementation of the constraint layer carries a precision error of 1e-08 due to a difference in dtypes (float64 in the original implementation and float32 in the presented implementation)
compared to [Eleonora Giunchiglia and Thomas Lukasiewicz implementation](https://github.com/EGiunchiglia/CCN).
Additionally, the presented implementation also divergences from the Giunchiglias implementation by adopting the general Constraint Module presented in
[4.2 of the journal article](https://doi.org/10.1613/jair.1.12850).
Furthermore, constraints are stored in a modified notatin replacing the old implication `:-` with `<-`.

# Model training

## Ablation study

A two-dimensional ablation study is run. Firstly, a study testing how much data is needed to train the model for a given configuration. This is done once in a series of increasing data subsets
for 10 epochs with a batch size of 32 and a learning rate of 1e-3. Taking inspiration from [Komatsuzaki, A. (2019).](https://arxiv.org/abs/1906.06669) a second model with the same
batch size and learning rate is run on the complete dataset for at maximum 1 epoch until metrics stagnate for an extended period.
Secondly, a study is run evaluating the impact of the constraint layer. This is done by running the model on the same dataset and the same configuration with and without the constraint layer.

# A data science perspective 

From a data science perspective, the data is insufficiently treated, and the unstructured text introduces unnecessary noise and is therefore not justifiable.
The text should be used to extract structured features. The EDA should be expanded to consider broader predictive measures such as the Predictive Power Score.
Furthermore, the model is needlessly complex. The problem formulation should be converted into a multi-class problem using a power set transformation from a multi-label problem. 
Predictive features from the literature described in the [EDA](./documentation/EDA.md#literature-context-of-potential-features) should be considered in full.

## Feature engineering

Very little feature engineering has been performed. The current solution causes the feature space to explode. Reducing the input dimension would almost certainly result in a better model.
No identification of correlation clusters has been performed, correlation clustering most certainly would improve the model.
The feature space could be significantly improved by grouping profiles into correlated clusters and segmenting across predictive features and removing redundently correlated features.
Additionally, new features such as past average turnover time per job could be a promising candidate as a predictor.

Elimination of the unstructured text would allow the use of SMOTE to oversample the dataset, thus providing a better way to solve the class imbalance problem.

It is unclear how well the missing data indicator is encoded in BERT by simply concatenating empty strings with SEP Tokens. Additionally, no predictiveness for missing data indicators has 
not been performed. If missing data in a variable is predictive it should be explicitly encoded and broken down by variable. One-hot encoded columns would automatically encode missing data.

On a procedural level, the explored but non-predictive features should be included in the EDA.

## Model adaptations

[Bogatinovski, J., Todorovski, L., Džeroski, S., & Kocev, D. (2022)](https://doi.org/10.1016/j.eswa.2022.117215) provided evidence that both random forest models specifically [RFPCT](https://doi.org/10.1016/j.patcog.2012.09.023)
and gradient-boosting models could be suitable replacements for the prediction head.
[Tanha, J., Abdi, Y., Samadi, N., Razzaghi, N., & Asadpour, M. (2020)](https://doi.org/10.1186/s40537-020-00349-y) found [Catboost](https://doi.org/10.48550/arXiv.1810.11363) to be the best-performing model amongst the boosting models.
However, Catboost unlike RFPCT does not account for label dependencies out of the box. Additionally, if the label dependence findings of Bogatinovski, Jasmin, et al. are credible new features capturing the
interdependence between labels should be introduced. Furthermore, [Moyano, J. M., Gibaja, E. L., Cios, K. J., & Ventura, S. (2018)](https://doi.org/10.1016/j.inffus.2017.12.001) found that a bagging Ensemble of Label Powerset classifiers (ELP)
performed the best for highly dependent label sets. However, ELP does not address the increase in complexity due to the power set generation or the dataset imbalance.

