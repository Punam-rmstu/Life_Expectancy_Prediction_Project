# Dataset Description: Life Expectancy Prediction

## **Source**
The dataset used in this project was collected from [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who), originally sourced from:
- **World Health Organization (WHO):** Health-related data for 193 countries.
- **United Nations (UN):** Economic data corresponding to the same countries.

This dataset provides comprehensive information about health, socio-economic, and environmental factors influencing life expectancy worldwide.

---

## **Dataset Overview**
The dataset spans **15 years (2000–2015)** and includes information for **193 countries**. It consists of critical variables that represent key health, economic, and social factors. After merging and cleaning, the final dataset contains:
- **Rows:** 2938
- **Columns:** 22 (including the target variable and 20 predictors)

---

## **Variable Categories**
The predicting variables are grouped into the following categories for better analysis:
- **Immunization-related factors:** 
  - Hepatitis B coverage
  - Polio coverage
  - DPT coverage
- **Mortality factors:**
  - Adult mortality rates
  - Infant deaths
  - HIV/AIDS prevalence
- **Economic factors:**
  - Gross Domestic Product (GDP)
  - Income composition of resources
- **Social factors:**
  - Alcohol consumption
  - Years of schooling
  - Population data

---

## **Data Preprocessing**
- **Merging and Cleaning:** Data from WHO and UN was merged into a single dataset.
- **Handling Missing Values:** 
  - Missing values were identified using the `Missmap` command in R.
  - Most missing data was observed for variables such as:
    - Population
    - Hepatitis B coverage
    - GDP
  - Missing data predominantly belonged to smaller countries like Vanuatu, Tonga, Togo, and Cabo Verde. These countries were excluded due to incomplete information.
- **Final Dataset:** 
  - The cleaned dataset was prepared with no significant errors.
  - It consists of 22 variables ready for predictive analysis.

---

## **Key Observations**
- Significant improvements in health outcomes have been observed globally, especially in developing nations, over the past 15 years compared to the previous three decades.
- The data highlights critical factors influencing life expectancy, such as healthcare access, economic stability, and social behaviors.

---

## **Usage**
This dataset can be used for:
- **Health Data Analysis:** Understanding trends and factors affecting life expectancy.
- **Predictive Modeling:** Developing machine learning models for life expectancy prediction.
- **Policy Development:** Guiding public health interventions based on insights derived from the data.

---

## **Acknowledgments**
We extend our gratitude to:
- The **World Health Organization (WHO)** for providing health data.
- The **United Nations (UN)** for economic data.
- [Kaggle]([https://www.kaggle.com/kumarajarshi/life-expectancy-who]) for making this dataset accessible to the public.

