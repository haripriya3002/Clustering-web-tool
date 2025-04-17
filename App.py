import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Wine Quality dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    return data

data = load_data()

# Streamlit app title and header
st.title(" Wine Quality Prediction")

# Display dataset information
st.subheader("Wine Quality Dataset")
st.dataframe(data)

# Select machine learning model
model_name = st.selectbox("Select a Machine Learning Model", ["Random Forest", "SVC", "Logistic Regression"])

# Split the data into features and target
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the selected model
if model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "SVC":
    model = SVC()
else:
    model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {accuracy:.2f}")

# Plot confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
ax.set_xlabel('Predicted Quality')
ax.set_ylabel('True Quality')
st.pyplot(fig)

# Add sidebar for additional features
st.sidebar.header("Visualization Options")
if st.sidebar.checkbox("Show Data Distribution"):
    st.subheader("Feature Distribution")
    feature = st.sidebar.selectbox("Select Feature to Visualize", data.columns[:-1])
    fig2, ax2 = plt.subplots()
    sns.histplot(data[feature], bins=30, kde=True, ax=ax2)
    ax2.set_title(f'Distribution of {feature}')
    st.pyplot(fig2)

# Additional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background: #f0f2f5;
    }
</style>
""", unsafe_allow_html=True)
