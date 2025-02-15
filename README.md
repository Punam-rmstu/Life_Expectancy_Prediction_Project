# **Life Expectancy Prediction Using Machine Learning**

## **Table of Contents**

1. [Contributors](#contributors)
2. [Instructor](#instructor)
3. [Course Information](#course-information)
4. [Project Overview](#project-overview)
   - [Objective](#objective)
   - [Significance](#significance)
   - [Dataset](#dataset)
   - [Key Features](#key-features)
   - [The Columns of the Dataset](#the-columns-of-the-dataset)
   - [Technologies Used](#technologies-used)
5. [Detailed Project Workflow](#detailed-project-workflow)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
   - [Model Training](#model-training)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Model Evaluation](#model-evaluation)
6. [Results and Insights](#results-and-insights)
7. [Recommendation](#recommendation)
8. [Future Enhancements](#future-enhancements)
9. [References](#references)

---
## **Contributors**

- **Punam Kanungoe**
  
  **Reg No**- 2018-15-24
  
  **Department**- CSE,RMSTU
  
- **Saikat Das Roky**
  
  **Reg No**- 2018-15-18
  
  **Department**- CSE,RMSTU
---  
## **Instructor**
  **Md Mynoddin**

  **Designation:** Assistant Professor
  
  **Department :** CSE, RMSTU

---
## Course Information  
- **Degree:** BSc in Computer Science and Engineering  
- **Year & Semester:** 4th Year, 2nd Semester  
- **Course Name & Code:** [Machine Learning Lab] (CSE-5211)

---
## **Project Overview**

This project focuses on predicting life expectancy using machine learning techniques by analyzing various socio-economic, demographic, and health-related factors. It aims to provide actionable insights into how these variables influence life expectancy across different regions and populations. By leveraging data-driven approaches, the project highlights critical patterns and trends that can support better decision-making for public health and policy initiatives.


### Objective

The primary objectives of this project are:

- To develop a machine learning model capable of accurately predicting life expectancy based on a wide range of factors, such as income, healthcare access, education, and lifestyle.
- To identify and quantify the impact of key features that influence life expectancy, offering insights into areas where intervention can improve public health outcomes.
- To provide governments, organizations, and businesses with actionable recommendations derived from the model to reduce disparities and improve quality of life globally.
  
### **Significance** 

Life expectancy is not only a measure of public health but also a reflection of a nation’s development and well-being. This project underscores its importance by:

- Equipping governments with data-driven insights to design effective public health policies.
- Empowering individuals with personalized health recommendations: By understanding the various factors that influence life expectancy, 
  individuals can be better informed about lifestyle choices that can improve their quality of life and longevity.
- Enabling businesses to assess market needs in sectors such as insurance, healthcare, and elder care.
- Providing a deeper understanding of how various social, economic, and health factors influence global life expectancy trends.
- Supporting global health initiatives: The project aids in identifying key factors contributing to life expectancy differences across regions, helping international organizations target interventions more effectively in areas with low life expectancy.
  

### **Dataset**
The dataset used in this project was collected from [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who), originally sourced from:
- **World Health Organization (WHO):** Health-related data for 193 countries.
- **United Nations (UN):** Economic data corresponding to the same countries.

This dataset provides comprehensive information about health, socio-economic, and environmental factors influencing life expectancy worldwide.
The dataset spans **16 years (2000–2015)** and includes information for **193 countries**. It consists of critical variables that represent key health, economic, and social factors. After merging and cleaning, the final dataset contains:
- **Rows:** 2938
- **Columns:** 22 (including the target variable and 20 predictors). 
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

- **Column Name Normalization**:  
  Standardized column names by converting them to lowercase and replacing spaces and special characters with underscores for consistency and ease of use in coding workflows.

- **Handling Missing Values**:  
  - For numeric features with missing values, applied median imputation to preserve robustness against outliers.  
  - Removed columns with excessive missing values (e.g., `hepatitis_b`) due to limited data availability and potential biases in the dataset. This ensured the reliability of the data used for modeling.  
  - Conducted a missing data heatmap analysis to identify patterns in missingness and ensure imputation strategies aligned with data trends.

- **Categorical Encoding**:  
  Encoded categorical variables:
  - `country` and `status` columns were transformed using `LabelEncoder` for numerical representation, enabling the models to process them effectively.

- **Feature Scaling**:  
  Applied `MinMaxScaler` to normalize numeric features into a uniform range (0-1). This scaling ensured that features with varying magnitudes contributed equally during model training, preventing bias toward large-valued features.
  

### **Exploratory Data Analysis (EDA)**

- **Distribution Analysis**:  
  Visualized distributions of key numerical features (e.g., life expectancy, GDP, BMI) using histograms and box plots to understand the spread, outliers, and skewness in the data.

- **Pairwise Relationships**:  
  Created pairplots to explore interactions between life expectancy and significant features, such as HIV/AIDS prevalence, BMI, and GDP. These visualizations revealed meaningful patterns, including nonlinear relationships.

- **Correlation Analysis**:  
  Generated a heatmap to identify feature correlations:  
  - Observed a strong negative correlation between adult mortality and life expectancy.  
  - Found positive correlations between life expectancy and features such as income composition of resources and schooling, indicating socioeconomic factors' critical influence.



### **Feature Engineering**

- **Derived Features**:  
  Engineered new features like `bmi_to_hiv_ratio` to capture the interplay between BMI and HIV/AIDS prevalence, providing additional predictive power to the model.  

- **Multicollinearity Reduction**:  
  Conducted variance inflation factor (VIF) analysis to detect and address multicollinearity among independent variables, ensuring model stability and interpretability.



### **Model Training**

Trained and evaluated the following regression models for life expectancy prediction:

- **Linear Regression**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **Support Vector Regressor (SVR)**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree Regressor**  

Models were selected to cover a mix of linear, ensemble, and non-linear approaches, providing a comprehensive evaluation of their performance.



### **Hyperparameter Tuning**

- Utilized `GridSearchCV` to fine-tune hyperparameters for models like Random Forest, Gradient Boosting, and SVR to achieve optimal performance.  
- Employed time-aware cross-validation (expanding window method) to mimic real-world scenarios, ensuring no data leakage and realistic evaluation metrics.



### **Model Evaluation**

- **Performance Metrics**:  
  Compared models using training and testing R² scores to evaluate how well the models fit the data and generalize to unseen samples.  

- **Residual Analysis**:  
  Analyzed residuals for patterns and inconsistencies, validating model assumptions and ensuring unbiased predictions.

- **Feature Importance**:  
  Conducted feature importance analysis using SHAP (SHapley Additive exPlanations) and permutation importance to understand the contribution of each feature to model predictions.
  
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

---

## **Recommendation**

Based on the insights from this project, the following actions are recommended:

- Governments should prioritize investments in healthcare infrastructure to improve access to medical services, particularly in underserved regions.
  
- Implement targeted programs to reduce the prevalence of diseases like HIV/AIDS by increasing awareness, providing preventive care, and ensuring affordable treatment options.  

- Enhance the quality of education systems and ensure equitable access to schooling, particularly in low-income and rural areas, as education has a significant impact on life expectancy.

- Develop policies aimed at reducing income inequality, as economic stability plays a critical role in improving overall health outcomes and life expectancy.

- Encourage public health campaigns focused on preventive care, healthy lifestyles, and early detection of chronic diseases to reduce mortality rates and enhance quality of life.

---

## **Future Enhancements**
1. Dataset Expansion: Incorporate more demographic, environmental, and genetic factors.
2. Real-time Integration: Develop an API-based real-time prediction system.
3. Explainability: Enhance model interpretability using SHAP or LIME.
4. Global Analysis: Extend the study to include datasets from diverse countries.
5. Advanced Techniques: Explore deep learning models for improved accuracy.

---

## **References**
- **WHO Life Expectancy Dataset**: [Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **https://developer.ibm.com/learningpaths/learning-path-machine-learning-for-developers**
- **Mahumud, R.A., Hossain, G., Hossain, R., Islam, N. and Rawal, L.,;, “Impact of Life Expectancy on Economic Growth and Health Care 
    Expenditures in Bangladesh.,” Universal Journal of Public Health, vol. 1, no. 4, pp. 180-186, 2013.**
- **Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.**
- **Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.**
  



