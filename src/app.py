import streamlit as st
import pickle
import numpy as np

# Loading the trained decision tree model
model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))

# Loading the label encoders
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Define the list of categorical features
categorical_features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color'
                        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring'
                        'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
                        ]

st.title('Mushroom Classification Prediction')

# Collect inputs using text input for feature options
input_features = []
for feature in categorical_features:
    user_input = st.text_input(f"Enter {feature.replace('_', ' ').title()} value", '')
    input_features.append(user_input)

# Predict button
if st.button('Predict'):
    # Encode the user inputs using the label encoders
    encoded_features = []
    for feature, user_input in zip(categorical_features, input_features):
        encoder = label_encoders[feature]
        try:
            # Here, we convert the user input to the encoded form
            encoded_input = encoder.transform([user_input])[0]
            encoded_features.append(encoded_input)
        except ValueError:
            st.error(f"Invalid input for {feature}")
            return
    
    # Now encoded_features contains the encoded categorical variables
    features_vector = np.array(encoded_features).reshape(1, -1)
    
    # Predict the class using the encoded features
    prediction = model.predict(features_vector)
    class_label = 'Poisonous' if prediction[0] == 1 else 'Edible'
    
    # Display the prediction result
    st.subheader(f'The mushroom is {class_label}')
