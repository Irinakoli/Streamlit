import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import itertools

st.header("Prediction of bank churn :money_with_wings:", divider = "rainbow")
st.subheader(":blue[Supervised] classification task")

st.markdown("The project aims to classify the features of the clients which make them likely  "
            "to resign from bank services.")

FILE = "Bank Customer Churn Prediction.csv"

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

df = load_data(file = FILE)
st.write(df[:5])


st.markdown(":red[credit_score] :blue[:left_right_arrow:] the trust score, the higher the better  "
            "\n:red[country] :blue[:left_right_arrow:] France, Spain or Germany  "
            "\n:red[gender] :blue[:left_right_arrow:] Female or Male  "
            "\n:red[tenure] :blue[:left_right_arrow:] years of being bank's customer  "
            "\n:red[credit_card] :blue[:left_right_arrow:] 1 : having credit card, 0: not having credit card  "
            "\n:red[active_member] :blue[:left_right_arrow:] 1: eager user, 0: passive user  "
            "\n:red[churn] :blue[:left_right_arrow:] 1: churned, 0: stayed")

st.divider()


labels = ["churned", "stayed"]
values = df.churn.value_counts(normalize = True).round(1)
pull_values = [0.2, 0]

fig = go.Figure(data=[go.Pie(labels= labels,
                             values= values,
                             pull= pull_values)])

st.plotly_chart(fig)

st.caption("In available dataset the customers who churned are the minority class ...")
st.caption("However, it is crucial to explore which features make the customer prone to :red[churn].")


st.header("**Exploratory Data Analysis**")

cols = ["credit_score", "age", "tenure", "balance",
        "products_number", "estimated_salary",
        "gender", "country", "churn"]

selected_feature = st.selectbox("Select a feature:", cols)
fig = px.histogram(df, x=selected_feature,
                   color="churn",
                   title=f"Countplot of {selected_feature} by churn")

st.plotly_chart(fig)

st.markdown("**Churned customers**: "
            "\n 1) are older  "
            "\n 2) have higher balance available on their accounts  "
            "\n 3) are passive users  "
            "\n 4) have much higher salary  "
            "\n 5) are mostly females  "
            "\n 6) are mostly from Germany  "
            "\n 7) have more than 3 products  "
            "\n 8) have credit_score below 400")


#preprocessing of categorical columns
#gender_dummies = pd.get_dummies(df.gender)
#country_dummies = pd.get_dummies(df.geography)
#df_new = pd.concat([df, gender_dummies, country_dummies], axis=1)
#df_new.drop(columns = ["gender", "geography"], inplace = True)

#scalling
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#df_new[['balance','estimatedsalary']]=scaler.fit_transform(df_new[['balance','estimatedsalary']])

from sklearn.preprocessing import MinMaxScaler

#min_scaler= MinMaxScaler()
#df_new[['creditscore']]=min_scaler.fit_transform(df_new[['creditscore']])

# imbalanced target variable handling

#cols = ["creditscore", "age", "tenure", "balance", "numofproducts", "hascrcard", "isactivemember", "estimatedsalary",
       #"exited", "Female", "Male", "France", "Germany", "Spain"]

#X = df.drop(columns = "exited", axis = 1)
#y = df.exited

#handling imbalanced dataset
#from imblearn.over_sampling import RandomOverSampler

#oversampler = RandomOverSampler(sampling_strategy='minority')
#X_balanced,y_balanced = oversampler.fit_resample(X,y)
#X_scaled = scaler.fit_transform(X_balanced)

#X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_balanced, test_size=0.3, random_state=45)

#model training
#rf = RandomForestClassifier()
#model = rf.fit(X_train, y_train)
#score = model.score(X_test, y_test).round(2)


gender_mapping = {0: 'Female', 1: 'Male'}
country_mapping = {0: 'France', 1: 'Germany', 2: 'Spain'}

def encode_categorical_columns(data):
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])
    return data

# Streamlit app
st.title("Churn Prediction App")

selected_categorical = st.multiselect("Select categorical columns:", df.select_dtypes(include=['object']).columns)

# Define binary options
binary_options = ["active_member", "credit_card"]

# User input for binary columns
selected_binary = []
for option in binary_options:
    selected_value = st.radio(f"Select {option}:", ["No", "Yes"], index=1)  # Labels "No" and "Yes"
    selected_binary.append(1 if selected_value == "Yes" else 0)

# User input for numerical columns with specified ranges
numerical_options = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                     if col not in ['churn', 'customer_id', "active_member", "credit_card"]]

selected_numerical = st.multiselect("Select numerical columns with specified ranges:", numerical_options)
range_dict = {}
for col in selected_numerical:
    min_val, max_val = st.slider(f"Select {col} range:", min(df[col]), max(df[col]), (min(df[col]), max(df[col])))
    range_dict[col] = (min_val, max_val)

# Apply filters based on user selections
filtered_df = df.copy()

for col in selected_categorical:
    original_values = df[col].unique()
    selected_values = st.multiselect(f"Select {col}:", original_values)
    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

filtered_df = encode_categorical_columns(filtered_df)  # Apply encoding to categorical columns

for col, selected_value in zip(binary_options, selected_binary):
    filtered_df = filtered_df[filtered_df[col] == selected_value]

for col, (min_val, max_val) in range_dict.items():
    filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]


# Function to calculate churn probabilities for a given set of features
def calculate_average_churn_probability(data):
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])

    X = data.drop('churn', axis=1)
    y = data['churn']

    clf = RandomForestClassifier()
    clf.fit(X, y)

    churn_probs = clf.predict_proba(X)[:, 1] * 100
    average_churn_probability = churn_probs.mean()  # Calculate the mean of churn probabilities
    return average_churn_probability

# Calculate average churn probability for the filtered data
average_churn_probability = calculate_average_churn_probability(filtered_df).round(2)

# Display the average churn probability
st.subheader("Average Churn Probability (%)")
st.write(average_churn_probability)
