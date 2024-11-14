# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load the dataset
data = pd.read_csv('/drive/MyDrive/Advertising.csv')

# Binarize the target variable (assuming sales > 15 is "High", else "Low")
data['Sales'] = data['Sales'].apply(lambda x: 'High' if x > 15 else 'Low')

# Select features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the GaussianNB model
model = GaussianNB()
model.fit(X_train, y_train)

# Save the model with pickle
with open('advertising_gaussian_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the trained model
with open('advertising_gaussian_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the Streamlit App
st.title('Advertising Sales Category Prediction')

# Description
st.write("Set values for TV, Radio, and Newspaper budgets, and predict whether sales will be 'High' or 'Low'.")

# Sliders for input features
tv = st.slider('TV Advertising Budget', min_value=0, max_value=300, step=1)
radio = st.slider('Radio Advertising Budget', min_value=0, max_value=50, step=1)
newspaper = st.slider('Newspaper Advertising Budget', min_value=0, max_value=100, step=1)

# Predict button
if st.button('Predict Sales Category'):
    # Prepare input for prediction
    input_features = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Display the prediction
    st.write(f"Predicted Sales Category: {prediction}")
