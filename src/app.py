import streamlit as st
import pickle
import numpy as np



# Load the trained decision tree model
model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))

# Define the list of features and their ranges
features_with_ranges = [
    ('cap-shape', 0, 5), ('cap-surface', 0, 3), ('cap-color', 0, 9), ('bruises', 0, 1), ('odor', 0, 8),
    ('gill-attachment', 0, 1), ('gill-spacing', 0, 1), ('gill-size', 0, 1), ('gill-color', 0, 11),
    ('stalk-shape', 0, 1), ('stalk-root', 0, 4), ('stalk-surface-above-ring', 0, 3), ('stalk-surface-below-ring', 0, 3),
    ('stalk-color-above-ring', 0, 8), ('stalk-color-below-ring', 0, 8), ('veil-type', 0, 0), ('veil-color', 0, 3),
    ('ring-number', 0, 2), ('ring-type', 0, 4), ('spore-print-color', 0, 8), ('population', 0, 5), ('habitat', 0, 6)
]


# Title of the application
st.title('Mushroom Classification Prediction')

# Collect inputs
input_features = []
for feature, min_val, max_val in features_with_ranges:
    value = st.slider(feature.replace('_', ' ').title(), min_val, max_val, min_val)
    input_features.append(value)

# Predict button
if st.button('Predict'):
    # Prepare the feature vector for prediction
    features_vector = np.array(input_features).reshape(1, -1)
    
    # Predict the class
    prediction = model.predict(features_vector)
    class_label = 'Poisonous' if prediction[0] == 1 else 'Edible'
    
    # Display the prediction result
    st.subheader(f'The mushroom is {class_label}')