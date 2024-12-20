# Churn-analysis


![image](https://github.com/user-attachments/assets/82a84da5-bf13-4383-a07a-b4c2fb95a432)


Customer churn is the phenomenon in which a client stops doing business with an entity. Users can stop using a company’s product or service for a variety of reasons, such as affordability, dissatisfaction with the offering, and bad customer service.

More often than not, customers who churn from one company will start doing business with their competitor. For instance, if you aren’t happy with your current mobile service provider due to slow Internet speed, you are likely to switch to an alternative.

The act of churning isn’t one that happens suddenly. If you experience low network bandwidth, you are likely to tolerate it for a month or two. During this period, you would probably contact customer support, check your network speed, and leave a review on social media expressing your dissatisfaction.

If the data scientists at your current provider can collect this data and ascertain that your behavior is similar to that of other customers who have churned in the past, they will immediately alert the marketing team, who will then reach out to you and attempt to cater to your needs in the best way possible. They may provide you with special promotions, upgrade your plan, and work on creating a satisfactory user experience for you to prevent you from leaving.

![image](https://github.com/user-attachments/assets/bbd1de9d-398a-4b02-92a2-fbfd1b6b5a33)


Customer churn prediction is one of the most popular use cases of data science in marketing. Companies incur a lot of costs when users churn since it is expensive to replace an existing customer. Due to this, most mid to large-sized organizations will have some sort of churn prediction mechanism in place.



![image](https://github.com/user-attachments/assets/5151eb6b-8cfe-4eb6-b7a0-9b18577f983b)

Offering streamlined experiences, competitive pricing, good service, and strong CRM contact center solutions are crucial for increasing customer retention. But being able to predict churn and address the customers' concerns timely is the key to nurturing loyal clients.

![image](https://github.com/user-attachments/assets/25dfa857-b3b3-4d51-9df2-31dc7764b629)


For subscription-based companies like Netflix and Spotify, it is crucial to retain existing customers since the entire business model relies on plan renewals. If you would like to work as a data scientist for companies like these in the future, it is a good idea to learn about techniques such as customer churn prediction. 

You can build a churn prediction model and showcase it on your resume, as this is a use case that is relevant to almost every organization and will help your portfolio stand out amongst other data science candidates.

![image](https://github.com/user-attachments/assets/46a8d3cd-bc24-4e95-8014-43d82b736205)


In this article, we will show you how to build a customer churn prediction model in Python using the random forests algorithm.

Let’s look into each attribute:

InternetService: It is clear from the visual above that customers who use fiber optic Internet churn more often than other users. This might be because fiber Internet is a more expensive service, or this provider doesn’t have good coverage.
TechSupport: Many users who churned did not sign up for tech support. This might mean that these customers did not receive any guidance on fixing technical issues and decided to stop using the service. 
OnlineBackup: Many customers who had churned did not sign up for an online backup service for data storage. 
Contract: Users who churned were almost always on a monthly contract. This makes sense, since these customers pay for the service on a monthly basis and can easily cancel their subscription before the next payment cycle.
Even without building a fancy machine learning model, a simple data-driven analysis like this can help organizations understand why they are losing customers and what they can do about it. 

For instance, if the company realizes that most of their users who churn have not signed up for tech support, they can include this as a complimentary service in some of their future product offerings to prevent other customers from leaving.

Step 4: Preprocessing Data for Customer Churn
Now that we have a better understanding of our dataset, let’s perform some data preparation before creating the machine learning model. There are three steps to this process:

Cleaning the dataset
Let’s look at the dataset summary again:

![image](https://github.com/user-attachments/assets/546b6853-56ad-4829-9dcd-babf86d1ae05)


Screenshot demonstration customer churn model in Jupyter

Notice that the variable “TotalCharges” has the data type “object,” when it should be a numeric column. Let’s convert this column into a numeric one:


Encoding Categorical Variables
The categorical variables in the dataset need to be converted into a numeric format before we can feed them into the machine learning model. We will perform the encoding using Scikit-Learn’s label encoder.

First, let’s take a look at the categorical features in the dataset:


Now, let’s take a look at the dataset after encoding these categorical variables:

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()


Notice that all the categorical values in the dataset have now been replaced with numbers.

Finally, run the following lines of code to merge the dataframe we just created with the previous one:

num_features = df[['customerID','TotalCharges','MonthlyCharges','SeniorCitizen','tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)
Oversampling
As mentioned above, the dataset is imbalanced, which means that a majority of values in the target variable belong to a single class. Most customers in the dataset did not churn - only 27% of them did.

This class imbalance problem can lead to an underperforming machine learning model. Some algorithms that train on an imbalanced dataset always end up predicting the majority class. In our case, for instance, the model may predict that none of the customers churned. While a model like this will be highly accurate (in this case it will be correct 73% of the time), it is of no value to us since it is always predicting a single outcome.

There are a variety of techniques that can be used to overcome the class imbalance problem in machine learning. In this tutorial, we will use a technique called oversampling. This is a process that involves randomly selecting samples from the minority class and adding it to the training dataset. We are going to oversample the minority class until the number of data points are equal to that of the majority class.

Before we oversample, let’s do a train-test split. We will oversample solely on the training dataset, as the test dataset must be representative of the true population:

choose which  model is best 

![image](https://github.com/user-attachments/assets/ea270eb4-9168-45e9-81f1-2b882f44d5e8)



Random Forest often performs well in classification tasks, including churn prediction, due to several advantages in handling complex datasets like the Telco Customer Churn dataset. Here’s why Random Forest might yield the highest predictive accuracy with this dataset:

1.Ensemble Method Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions. By aggregating the output from numerous decision trees, it reduces the risk of overfitting, which can be common with individual decision trees. This approach typically improves the robustness and generalizability of the model, resulting in better performance on new data.

2.Handling Non-Linear Relationships The Telco Customer Churn dataset likely contains non-linear relationships between features (e.g., Monthly Charges, Tenure, Payment Method) and the target variable (Churn). Random Forest, being a collection of decision trees, is naturally adept at capturing these non-linear patterns without requiring explicit feature transformations or engineering.

3.Feature Importance and Automatic Selection Random Forest assigns importance to each feature based on how well it contributes to the prediction. This means it automatically gives more weight to the most important features, potentially filtering out irrelevant features that might cause noise. For churn prediction, features like MonthlyCharges, Contract, Tenure, and PaymentMethod are likely significant, and Random Forest can leverage them effectively.

4.Handles Missing Values and Categorical Data Decision trees within the Random Forest can handle missing data and categorical variables relatively well. In many cases, Random Forests can split data based on subsets of values in categorical features, which can be useful when dealing with features like InternetService or PaymentMethod in the Telco dataset.

Works Well with Imbalanced Data If the dataset is imbalanced (e.g., fewer customers churn compared to those who do not), Random Forest can still perform effectively. It achieves this by training multiple trees that capture the minority class better, thus improving predictive performance on both churned and non-churned customers.

5.Reduced Overfitting Random Forest’s use of random feature selection and multiple trees significantly reduces overfitting compared to a single decision tree. Overfitting is especially problematic with complex datasets, where a model might memorize training data rather than learn patterns. By averaging out the predictions, Random Forest achieves a balance between bias and variance, leading to better generalization.

6.Robust to Outliers and Noise Random Forests are relatively robust to outliers because each tree is built on a different bootstrap sample and may not include the outliers. Outliers in monthly charges, tenure, or other variables in the Telco dataset, for instance, may not affect all trees, thus reducing their impact on the overall model performance.

