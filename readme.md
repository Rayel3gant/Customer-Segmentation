# Customer Segmentation
Customer Segmentation involves grouping customers based on shared characteristics, behaviors and preferences. By segmenting customers, businesses can tailor their strategies and target specific groups more effectively and enhance overall market value.

## Feature Description

| Feature | Meaning |
|---|---|
| **ID** | Unique identifier for each customer. Usually **not useful for clustering**. |
| **Year_Birth** | Customer's birth year. Can be transformed into **Age**. |
| **Education** | Customer's education level, e.g. Graduation, Master, PhD. |
| **Marital_Status** | Customer's marital/family status. |
| **Income** | Customer's annual household income. |
| **Kidhome** | Number of children in the household. |
| **Teenhome** | Number of teenagers in the household. |
| **Dt_Customer** | Date when the customer enrolled/registered with the company. Can be used to calculate **customer tenure**. |
| **Recency** | Number of days since the customer's last purchase. **Lower = more recently active.** |
| **MntWines** | Amount spent on wine products. |
| **MntFruits** | Amount spent on fruit products. |
| **MntMeatProducts** | Amount spent on meat products. |
| **MntFishProducts** | Amount spent on fish products. |
| **MntSweetProducts** | Amount spent on sweet products. |
| **MntGoldProds** | Amount spent on gold/other premium products. |
| **NumDealsPurchases** | Number of purchases made using a discount/deal. |
| **NumWebPurchases** | Number of purchases made through the company's website. |
| **NumCatalogPurchases** | Number of purchases made through the catalog. |
| **NumStorePurchases** | Number of purchases made directly in physical stores. |
| **NumWebVisitsMonth** | Number of visits to the company's website per month. |
| **AcceptedCmp3** | Whether the customer accepted Campaign 3. `1 = Yes`, `0 = No`. |
| **AcceptedCmp4** | Whether the customer accepted Campaign 4. `1 = Yes`, `0 = No`. |
| **AcceptedCmp5** | Whether the customer accepted Campaign 5. `1 = Yes`, `0 = No`. |
| **AcceptedCmp1** | Whether the customer accepted Campaign 1. `1 = Yes`, `0 = No`. |
| **AcceptedCmp2** | Whether the customer accepted Campaign 2. `1 = Yes`, `0 = No`. |
| **Complain** | Whether the customer complained in the last 2 years. `1 = Yes`, `0 = No`. |
| **Z_CostContact** | Fixed/placeholder value related to the cost of contacting a customer.|
| **Z_Revenue** | Fixed/placeholder value related to revenue.|
| **Response** | Whether the customer accepted the **most recent campaign**. `1 = Yes`, `0 = No`. |


## Run Locally

Clone the project

```bash
  git clone https://github.com/Rayel3gant/Customer-Segmentation
```

Go to the project directory

```bash
  cd Customer-Segmentation
```

Install dependencies

```bash
  pip install pandas, numpy , seaborn , matplotlib, scipy , scikit-learn , FastAPI, uvicorn, pickle, boto3, certifi, pymongo , evidently, 
```

Environment Variables
```bash
  AWS_ACCESS_KEY_ID_ENV_KEY = 
  AWS_SECRET_ACCESS_KEY_ENV_KEY = 
  MONGODB_URL_KEY = 
```

AWS Setup

1. Create an **AWS account** through the AWS Management Console.
2. Create an **IAM user** and grant the required **Amazon S3 permissions** for the project.
3. Generate and securely store the IAM user's **Access Key ID** and **Secret Access Key**.
4. Create an **S3 bucket** using the bucket name configured in the project: `TRAINING_BUCKET_NAME`


Run Code
```bash
  python app.py
```

## Conclusions

* Performed **outlier analysis** to identify extreme values in numerical features and **correlation analysis** to detect highly correlated features.
* Since the dataset was relatively small, outliers were not completely removed during the EDA stage to avoid losing potentially valuable customer information.
* Ingested customer data from **MongoDB Atlas** and validated the training and test datasets using **data drift analysis**.
* Implemented **feature engineering** to create meaningful customer profile features and reduce feature redundancy.
* Applied appropriate data transformations, including **StandardScaler** and **PowerTransformer**, to improve the suitability of features for modeling.
* Used **KMeans Clustering** to segment customers into distinct customer groups and generate cluster labels.
* Evaluated multiple supervised learning algorithms to identify a suitable model for predicting customer clusters. The **Model Factory** internally performs **GridSearchCV** for hyperparameter tuning and selects the best-performing model.
* Achieved approximately **85% test accuracy** . Serialized the trained model as a **pickle file** and stored it in an **AWS S3 bucket** for customer cluster prediction through the deployed web form.

