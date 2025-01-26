# **Life Expectancy Prediction Using Machine Learning**

## **Table of Contents**

1. [Contributors and Instructor](#contributors-and-instructor)
2. [Project Overview](#project-overview)
   - [Objective](#objective)
   - [Significance](#significance)
   - [Dataset](#dataset)
   - [Key Features](#key-features)
   - [The Columns of the Dataset](#the-columns-of-the-dataset)
   - [Technologies Used](#technologies-used)
3. [Detailed Project Workflow](#detailed-project-workflow)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
   - [Model Training](#model-training)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Model Evaluation](#model-evaluation)
4. [Results and Insights](#results-and-insights)
5. [Future Enhancements](#future-enhancements)
6. [References](#references)

---
## **Contributors and Instructor**
### **Contributors**
- **Punam Kanungoe**
  
  **Reg No**- 2018-15-24
  
    Department of CSE,RMSTU
  
- **Saikat Das Roky**
  
  **Reg No**- 2018-15-18
  
    Department of CSE,RMSTU

### **Instructor**
- **Md Mynoddin**

  Assistant Professor , Department of CSE, RMSTU

---
## **Project Overview**

Life expectancy is a key indicator of a nation's health, socio-economic progress, and overall quality of life. Understanding and accurately predicting life expectancy provides insights into the combined effects of health, environmental, and economic factors, enabling targeted and effective public health strategies.

This project leverages machine learning techniques to analyze global datasets, identifying critical predictors such as healthcare access, income levels, disease prevalence, and lifestyle choices. These models uncover patterns and trends that traditional methods might overlook.

The study aims to provide policymakers with actionable, data-driven insights to address health disparities and improve life expectancy, especially in resource-limited regions.


### **Objective**
The goal of this project is to build a reliable machine learning pipeline capable of predicting life expectancy based on various health, demographic, and socioeconomic factors. By analyzing critical predictors, the project aims to uncover key drivers of life expectancy, providing actionable insights to improve public health strategies and optimize resource allocation. Additionally, the study emphasizes feature importance to guide policymakers in prioritizing impactful interventions for enhancing global health outcomes.


### **Significance** 

Life expectancy is not only a measure of public health but also a reflection of a nation’s development and well-being. This project underscores its importance by:

- Equipping governments with data-driven insights to design effective public health policies.
- Empowering individuals with personalized health recommendations: By understanding the various factors that influence life expectancy, 
  individuals can be better informed about lifestyle choices that can improve their quality of life and longevity.
- Guiding organizations in addressing disparities in healthcare access and outcomes.
- Enabling businesses to assess market needs in sectors such as insurance, healthcare, and elder care.
- Providing a deeper understanding of how various social, economic, and health factors influence global life expectancy trends.
- Supporting global health initiatives: The project aids in identifying key factors contributing to life expectancy differences across regions, helping international organizations target interventions more effectively in areas with low life expectancy.

### **Dataset**
The dataset used in this project was collected from [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who), originally sourced from:
- **World Health Organization (WHO):** Health-related data for 193 countries.
- **United Nations (UN):** Economic data corresponding to the same countries.

This dataset provides comprehensive information about health, socio-economic, and environmental factors influencing life expectancy worldwide.
The dataset spans **15 years (2000–2015)** and includes information for **193 countries**. It consists of critical variables that represent key health, economic, and social factors. After merging and cleaning, the final dataset contains:
- **Rows:** 2938
- **Columns:** 22 (including the target variable and 20 predictors).
- 
- Health, demographic, and socioeconomic indicators, such as:
  - Life expectancy (target variable)
  - Adult mortality, infant deaths
  - BMI, schooling, GDP, immunization coverage
  - HIV/AIDS prevalence, thinness among children

### **Key Features**
  - **Target Variable**: Life expectancy (in years)
  - **Demographic**: Population, schooling, income composition
  - **Health Indicators**: BMI, HIV/AIDS prevalence, immunization coverage
  - **Economic**: GDP, health expenditure
  - **Mortality Rates**: Adult, infant, and under-five mortality

### **The Columns of the Dataset**
  - **Country** : Country
  - **Year** : Year
  - **Status** : Country Developed or Developing status
  - **Life expectancy** : Life expectancy in age
  - **Adult Mortality** : Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
  - **infant deaths** : Number of Infant Deaths per 1000 population
  - **Alcohol** : Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol) -percentage expenditure: Expenditure on 
                  health as a percentage of Gross Domestic Product per capita(%)
  - **Hepatitis B** : Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
  - **Measles** : Measles - number of reported cases per 1000 population
  - **BMI** : Average Body Mass Index of entire population
  - **under-five deaths** : Number of under-five deaths per 1000 population
  - **Polio** : Polio (Pol3) immunization coverage among 1-year-olds (%)
  - **Total expenditure** : General government expenditure on health as a percentage of total government expenditure (%)
  - **Diphtheria** : Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
  - **HIV/AIDS** : Deaths per 1 000 live births HIV/AIDS (0-4 years)
  - **GDP** : Gross Domestic Product per capita (in USD)
  - **Population** : Population of the country
  - **thinness 1-19 years** : Prevalence of thinness among children and adolescents for Age 10 to 19 (%)
  - **thinness 5-9 years** : Prevalence of thinness among children for Age 5 to 9(%)
  - **Income composition of resources** : Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
  - **Schooling** : Number of years of Schooling(years)

### **Technologies Used**
- **Programming Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, SHAP
- **Notebook Environment**: Google Colab / Jupyter Notebook

---

## **Detailed Project Workflow**

### **Data Preprocessing**
- **Column Name Normalization**: Simplified column names by converting them to lowercase and replacing spaces with underscores.
- **Handling Missing Values**:
  - Imputed missing values for numeric features using median imputation.
  - Dropped unreliable columns such as Hepatitis B due to high missing values and suspected bias.
- **Categorical Encoding**:
  - Transformed `country` and `status` columns using `LabelEncoder`.
- **Feature Scaling**:
  - Scaled numeric features using `MinMaxScaler` to standardize the range for consistent model performance.

### **Exploratory Data Analysis (EDA)**
- Analyzed the distributions of key numeric variables using histograms.
- Visualized pairwise relationships among features using pairplots (e.g., life expectancy vs. HIV/AIDS prevalence, BMI, and GDP).
- Generated a correlation heatmap to uncover strong relationships, such as:
  - Negative correlation between adult mortality and life expectancy.
  - Positive correlation between income composition of resources, schooling, and life expectancy.

### **Feature Engineering**
- Engineered a derived feature: `bmi_to_hiv_ratio` for better representation of the relationship between BMI and HIV/AIDS prevalence.
- Focused on reducing multicollinearity among features.

### **Model Training**
Trained and evaluated the following regression models:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Regressor**

### **Hyperparameter Tuning**
- Used `GridSearchCV` to optimize hyperparameters for models like Random Forest, Gradient Boosting, and SVR.
- Applied time-aware cross-validation (expanding window method) to prevent data leakage and ensure realistic performance evaluation.

### **Model Evaluation**
- Compared models using:
  - **Training R²**: How well the model fits the training data.
  - **Testing R²**: How well the model generalizes to unseen data.
- Analyzed residual distributions for model validation.
- Conducted feature importance analysis using SHAP and permutation importance.

---

## **Results and Insights**

### **Best Model**
- **Gradient Boosting Regressor**:
  - **Training R²**: 0.9875
  - **Testing R²**: 0.9593

### **Key Predictors**
- **HIV/AIDS Prevalence**: Strongest negative impact on life expectancy.
- **Income Composition of Resources**: Indicates the effectiveness of income utilization for human development.
- **Adult Mortality**: Significant negative correlation with life expectancy.

### **Feature Importance**
- Socioeconomic and demographic features like schooling, GDP, and income composition emerged as the most influential predictors.
- Health indicators, such as immunization coverage, had relatively lower importance due to widespread implementation globally.

### **Recommendations**
- Governments should focus on:
  - Reducing HIV/AIDS prevalence.
  - Improving education systems to ensure equitable access to quality schooling.
  - Increasing investments in health infrastructure and income equality.

---

## **Future Enhancements**
1. Incorporate additional datasets (e.g., global economic indices) for enhanced predictions.
2. Deploy the model via a Flask API for real-time predictions.
3. Implement ensemble methods (e.g., stacking) for improved accuracy.
4. Automate dataset preprocessing and feature engineering pipelines.

---


## **References**
- **WHO Life Expectancy Dataset**: [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.**
- **Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.**
- **Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.**




