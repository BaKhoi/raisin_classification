import pickle
import numpy as np


# Function to get user input
def get_input():
    
    # Input features
    features = ['Area', 'Major Axis Length', 'Minor Axis Length','Eccentricity', 'Convex Area', 'Extent', 'Perimeter']
    
    # User input list
    user_input = []
    
    # Get input
    for i in range(len(features)):
        feature = float(input(f"Enter feature {features[i]}: "))
        user_input.append(feature)
    return user_input



# Function to predict from input
def predict_raisin(model, user_input):
    # Convert user input to numpy array (example: [[feature1, feature2, ...]])
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

def convert_to_category(prediction):
    label_mapping = {0: "Kecimen", 1: "Besni"}
    return label_mapping[prediction[0]]


def main():
    
    # load the model from disk
    filename = 'raisin_classification_svm.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    
    # Get input from user
    input = get_input()
    
    # Predict 
    prediction = predict_raisin(loaded_model, input)
    print(f'\nThe features belong to {convert_to_category(prediction)} class')
    

if __name__ == "__main__":
    main()