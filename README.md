# **Customer Segmentation For An E-Commerce Website**
*Sofia Chevrolat (August 2020)*

> NB: This project is the 5th of a series of 7 projects comprising [the syllabus offered by OpenClassrooms in partnership with Centrale Supélec and sanctioned by the Data Scientist diploma - Master level](https://openclassrooms.com/fr/paths/164-data-scientist).
___

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.madimedia.pro)

This study in 2 notebooks aims segment Olist's clients by exploiting their anonymized customer database, containing information pertaining to their clients' purchase history, satisfactions and localization since january 2017.
___

This study is divided into 2 notebooks: 
- A cleaning up, feature engineering and EDA notebook
- A modeling and prediction notebook
___
## Notebook 1 : Data Clean Up, Feature Engineering and EDA

This notebook is organised as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries and functions
- 0.2 Loading and description of the dataset
- 0.3 Assembling the data
    
**1. Cleaning Up**
- 1.1 Spelling and typological corrections
- 1.2 Deleting outliers
    * 1.2.1 Deleting products with a weight indicated as <= 0
    * 1.2.2 Deleting results of workflow errors
- 1.3 Deleting the least filled in features
- 1.4 Deleting observations that are entirely empty
- 1.5 Deleting NaN values

**2. Data Targeting**

**3. Feature Engineering**
- 3.1 Creating new features
- 3.2 Encoding
    * 3.2.1 One Hot Encoding
    * 3.2.2 Feature Hashing
- 3.3 Reassembling the dataset

**4. EDA**
- 4.1 Statistical values
    * 4.1.1 Central tendency
        * 4.1.1.1 Qualitative features
        * 4.1.1.2 Quantitative features
    * 4.1.2 Feature distribution
        * 4.1.2.1 Qualitative features
        * 4.1.2.2 Quantitative features
- 4.2 Study of the correlations between the average basket and the RFM scores
    * 4.2.1 Qualitative variables
        * 4.2.1.1 With the average basket
        * 4.2.1.2 With the RFM scores
    * 4.2.2 Quantitative variables
        * 4.2.2.1 With the average basket
        * 4.2.2.2 With the RFM scores
- 4.3 Business analysis
    * 4.3.1 Client analysis
        * 4.3.1.1 Evolution of the number of clients and new clients
        * 4.3.1.2 Evolution of the distribution of clients per state
        * 4.3.1.3 Evolution of the distribution of the RFM scores
    * 4.3.2 Order analysis
        * 4.3.2.1 Evolution of the number of orders per year and per client
        * 4.3.2.2 Evolution of the pourcentage of orders in income per state
        * 4.3.2.3 Evolution of the distribution of the categories of orders per state

**5. Exporting the data**

**6. Conclusions**

___
## Notebook 2 : Data Modeling

This notebook is organized as follows:

**0. Setting Up**
- 0.1 Loading the necessary libraries and functions
- 0.2 Loading the dataset
- 0.3 Preparing the features
    * 0.3.1 Creating features using the dates of the orders
    * 0.3.2 Transformation to log
    * 0.3.3. Standardization

**1. Dimensionality Reduction**
- 1.1 PCA
    * 1.1.1 Choosing the number of components
    * 1.1.2 Latent variables explaining the data
- 1.2 t-SNE
    * 1.2.1 t-SNE projections
    * 1.2.2 t-SNE on PCA projections
- 1.3 Conclusions

**2. Segmentation**
- 2.1 Picking evaluation criteria
- 2.2 Algorithm selection
    * 2.2.1 Number of clusters a priori
    * 2.2.2 Perofmance comparison
    * 2.2.3 Conclusions
- 2.3 Parameter exploration
    * 2.3.1 Number of clusters
        * 2.3.1.1 Elbow method
        * 2.3.1.2 Average silhouette coefficient method
        * 2.3.1.3 Tests and comparisons with k taking on a range of values
    * 2.3.2 Other parameters
    * 2.3.3 Conclusions
- 2.4 Model use
    * 2.4.1 Segmentation with determination of the centroids
    * 2.4.2 Determination of the medoids
    * 2.4.3 Average client profile by cluster
- 2.5 Frequency analysis
    * 2.5.1 Test of the model on the data for year 2018
        * 2.5.1.1 Dimensionality reduction
        * 2.5.1.2 Segmentation and medoid determination
        * 2.5.1.3 Average client profile per cluster
    * 2.5.2 Comparison average client profile 2017 and 2018 per cluster
    * 2.5.3 Conclusions
    
**3. Final Segmentation**
- 3.1 Dimensionality reduction: t-SNE on PCA
- 3.2 Model set up
    * 3.2.1 Choosing the number of clusters
        * 3.2.1.1 Dendrogram
        * 3.2.1.2 Elbow method
    * 3.2.2 Optimizing the other parameters
- 3.3 Segmentation
    * 3.3.1 Applying the model
    * 3.3.2 Determining the medoids
    * 3.3.3 Average client profile per cluster
    
**4. Conclusions**

_________

## Requirements

This assumes that you already have an environment allowing you to run Jupyter notebooks. 

The libraries used otherwise are listed in requirements.txt

_________

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce), and place it under Resources/datasets/.

2. Run the following in your terminal to install all required libraries :

```bash
pip3 install -r requirements.txt
```

4. Run the notebooks in order (Notebook 1 first, then Notebook 2).
__________

## Results

For a complete presentation and commentary of the results of this analysis, please see the PowerPoint presentation.

> NB: The presentation is in French.
