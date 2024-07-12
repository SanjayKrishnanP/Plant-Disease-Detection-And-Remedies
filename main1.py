import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Load the pre-trained disease classification model
model = load_model(r"C:\Users\AISU\Downloads\ui _temp\plant_disease_prediction_model2.h5")

# Define a dictionary to map disease indices to disease names and remedies
disease_names = {
  
    0: ("Apple___Apple_scab", "Spray with copper-based fungicide."),
    1: ("Apple___Black_rot", "Apply fungicide containing chlorothalonil."),
    2: ("Apple___Cedar_apple_rust", "Remove any infected leaves, branches, or fruit from the area around the tree. This helps prevent the spread of the disease."),
    3: ("Apple___healthy", "No remedy needed."),
    4: ("Blueberry___healthy", "No remedy needed."),
    5: ("Cherry_(including_sour)___Powdery_mildew", "Apply fungicides labeled for powdery mildew control. Fungicides containing active ingredients like sulfur, potassium bicarbonate, neem oil, or horticultural oil can be effective against powdery mildew. Follow the instructions on the label carefully."),
    6: ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Apply fungicides labeled for control of cercospora leaf spot and gray leaf spot. Fungicides containing active ingredients such as azoxystrobin, pyraclostrobin, trifloxystrobin, or propiconazole can be effective. Follow the instructions on the label regarding application timing and rates."),
    7: ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Deep plowing after harvest can bury crop residue, reducing the survival of fungal spores in the soil. However, excessive tillage can lead to soil erosion and other environmental issues, so it's essential to balance tillage practices with soil conservation efforts."),
    8: ("Corn_(maize)___Common_rust_", "Apply fungicides labeled for control of common rust. Fungicides containing active ingredients such as azoxystrobin, pyraclostrobin, trifloxystrobin, or propiconazole can be effective. Follow the instructions on the label regarding application timing and rates."),
    9: ("Corn_(maize)___Northern_Leaf_Blight", "Crop Rotation: Rotate corn with non-host crops to break the disease cycle. Avoid planting corn in the same field or area where it was grown the previous year."),
    10: ("Corn_(maize)___healthy", "No remedy needed."),
    11: ("Grape___Black_rot", "Thin out grape clusters to reduce crowding and improve air circulation. This helps prevent moisture buildup and reduces the risk of black rot infection."),
    12: ("Grape___Esca_(Black_Measles)", "Prune grapevines during the dormant season, removing any diseased wood. Pruning can help reduce the disease's spread and improve the vine's overall health."),
    13: ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Apply fungicides labeled for control of Isariopsis leaf spot. Fungicides containing active ingredients such as captan, mancozeb, azoxystrobin, or tebuconazole may be effective. Follow the instructions on the label regarding application timing and rates. Begin spraying preventively when conditions favor disease development, such as during periods of wet weather."),
    14: ("Grape - Healthy", "Maintain proper vineyard management practices, including regular pruning, adequate irrigation, and balanced fertilization, to promote vine health and vigor."),
    15: ("Orange - Huanglongbing (Citrus greening)", "Implement aggressive vector control measures to manage the Asian citrus psyllid, the insect responsible for transmitting the bacteria that cause Huanglongbing. Remove and destroy infected trees to prevent the spread of the disease."),
    16: ("Peach - Bacterial spot", "Apply copper-based fungicides during the dormant season to manage bacterial spot in peach trees. Prune out infected branches and remove diseased fruit to reduce disease pressure."),
    17: ("Peach - Healthy", "Practice good orchard management, including proper pruning, irrigation, and fertilization, to promote peach tree health and productivity."),
    18: ("Pepper_Bell - Bacterial spot", "Apply copper-based fungicides preventively and rotate with bactericides containing streptomycin to manage bacterial spot in bell pepper plants. Practice crop rotation and sanitation to reduce disease pressure."),
    19: ("Pepper, Bell - Healthy", "Maintain optimal growing conditions, including adequate sunlight, water, and nutrient levels, to promote the health and productivity of bell pepper plants."),
    20: ("Potato - Early blight", "Apply fungicides containing chlorothalonil or mancozeb to manage early blight in potato plants. Rotate potato crops with non-host crops and practice good sanitation to reduce disease pressure."),
    21: ("Potato - Late blight", "Apply fungicides containing chlorothalonil or mancozeb preventively and rotate with other fungicides to manage late blight in potato plants. Remove and destroy infected plants to prevent the spread of the disease."),
    22: ("Potato - Healthy", "Practice crop rotation, proper planting depth, and irrigation management to promote healthy potato growth and reduce the risk of disease."),
    23: ("Raspberry - Healthy", "Monitor raspberry plants regularly for signs of pests and diseases and promptly treat any issues that arise. Maintain proper pruning and training to promote air circulation and reduce disease pressure."),
    24: ("Soybean - Healthy", "Practice crop rotation with non-host crops, use certified disease-free seeds, and monitor soybean fields regularly for signs of pests and diseases to maintain soybean health."),
    25: ("Squash - Powdery mildew", "Apply fungicides containing sulfur or potassium bicarbonate to manage powdery mildew in squash plants. Plant resistant varieties and maintain proper spacing to promote airflow and reduce humidity."),
    26: ("Strawberry - Leaf scorch", "Apply fungicides containing chlorothalonil or copper-based products to manage leaf scorch in strawberry plants. Remove and destroy infected leaves to reduce disease spread."),
    27: ("Strawberry - Healthy", "Plant disease-resistant strawberry varieties, maintain proper spacing between plants, and provide adequate irrigation and nutrition to promote healthy strawberry growth."),
    28: ("Tomato - Bacterial spot", "Apply copper-based fungicides preventively and rotate with bactericides containing streptomycin to manage bacterial spot in tomato plants. Practice crop rotation and sanitation to reduce disease pressure."),
    29: ("Tomato - Early blight", "Apply fungicides containing chlorothalonil or mancozeb to manage early blight in tomato plants. Rotate tomato crops with non-host crops and practice good sanitation to reduce disease pressure."),
    30: ("Tomato - Late blight", "Apply fungicides containing chlorothalonil or mancozeb preventively and rotate with other fungicides to manage late blight in tomato plants. Remove and destroy infected plants to prevent the spread of the disease."),
31: ("Tomato - Leaf Mold", "Apply fungicides containing chlorothalonil or copper-based products to manage leaf mold in tomato plants. Maintain proper spacing and pruning to promote airflow and reduce humidity."),
32: ("Tomato - Septoria leaf spot", "Apply fungicides containing chlorothalonil or mancozeb to manage Septoria leaf spot in tomato plants. Remove and destroy infected leaves and plant debris to reduce disease pressure."),
33: ("Tomato - Spider mites", "Apply insecticidal soaps or horticultural oils to manage spider mite infestations in tomato plants. Monitor plants regularly and remove heavily infested leaves to reduce mite populations."),
34: ("Tomato - Target spot", "Apply fungicides containing chlorothalonil or mancozeb to manage target spot in tomato plants. Remove and destroy infected plant debris to reduce disease pressure."),
35: ("Tomato - Tomato yellow leaf curl virus", "Control whiteflies, the vectors of tomato yellow leaf curl virus, using insecticides and reflective mulches. Remove and destroy infected plants to prevent the spread of the virus."),
36: ("Tomato - Tomato mosaic virus", "There is no cure for tomato mosaic virus. Plant resistant varieties, control aphids, and practice strict sanitation to reduce virus transmission."),
37: ("Tomato - Healthy", "Rotate tomato crops with non-host crops, practice proper irrigation and fertilization, and monitor plants regularly for signs of pests and diseases to maintain tomato health.")

    # Update other diseases and remedies similarly
}

def predict_disease(image):
    """
    Preprocesses and predicts disease from an uploaded image.

    Args:
        image: A PIL image object.

    Returns:
        A tuple containing the predicted disease name and corresponding remedy.
    """
    # Preprocess the image (resize, normalize etc.) based on your model's requirements
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize image to match model input shape
    img = img / 255.0  # Normalize pixel values to range [0, 1]

    # Predict disease class using the model
    predicted_disease_index = np.argmax(model.predict(np.expand_dims(img, axis=0)))

    # Retrieve disease name and remedy from dictionary
    predicted_disease, remedy = disease_names.get(predicted_disease_index, ("Unknown", "No remedy found"))

    return predicted_disease, remedy

def main():
    """
    Streamlit application entry point.
    """
    st.title("Plant Disease Prediction App")
    uploaded_file = st.file_uploader("Choose an image of your plant", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict Disease"):
            predicted_disease, remedy = predict_disease(image)
            st.success(f"Predicted Disease: {predicted_disease}")
            st.write(f"Remedy: {remedy}")

if __name__ == "__main__":
    main()

