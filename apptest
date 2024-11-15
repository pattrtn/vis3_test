import streamlit as st
import joblib
import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load CRF model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Function to process input text with the CRF model
def predict_entities(crf_model, text):
    tokens = text.split()  # Tokenize the input text
    features = [tokens_to_features(tokens, i) for i in range(len(tokens))]  # Generate features for each token
    predictions = crf_model.predict([features])[0]  # Predict entities for the tokens
    return list(zip(tokens, predictions))  # Return a list of token-entity pairs

# Feature extraction function
def tokens_to_features(tokens, i):
    stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]  # Define common stopwords
    word = tokens[i]  # Current token

    features = {
        "bias": 1.0,  # Bias term to improve model performance
        "word.word": word,
        "word[:3]": word[:3],  # First three characters of the word
        "word.isspace()": word.isspace(),
        "word.is_stopword()": word in stopwords,  # Check if stopword
        "word.isdigit()": word.isdigit(),
    }

    # Previous and next word features
    if i > 0:
        prevword = tokens[i - 1]
        features.update({
            "-1.word.prevword": prevword,
        })
    if i < len(tokens) - 1:
        nextword = tokens[i + 1]
        features.update({
            "+1.word.nextword": nextword,
        })
    return features

# Load data for dropdowns
file_path = './thaidata.xlsx'
data = pd.read_excel(file_path, sheet_name='db')
tambon_options = [""] + data['TambonThaiShort'].dropna().unique().tolist()
district_options = [""] + data['DistrictThaiShort'].dropna().unique().tolist()
province_options = [""] + data['ProvinceThai'].dropna().unique().tolist()

# Map postal codes to district, subdistrict, and province
postal_code_mapping = data.set_index(['TambonThaiShort', 'DistrictThaiShort', 'ProvinceThai'])['PostCodeMain'].to_dict()

# Streamlit app setup
st.title("NER Model Visualization")
st.markdown("""This app allows you to visualize and interact with a Named Entity Recognition (NER) model trained for address extraction. Input the required fields and run the model to see predictions.""")

# Load the CRF model from a predefined file path
model_file_path = './model.joblib'
model = load_model(model_file_path)  # Load the model
st.success("Model loaded successfully!")

# Input fields for address components
name = st.text_input("ชื่อ (Name):")
address = st.text_input("ที่อยู่ (Address):")
district = st.selectbox("ตำบล (District):", options=tambon_options)
subdistrict = st.selectbox("อำเภอ (Sub-district):", options=district_options)
province = st.selectbox("จังหวัด (Province):", options=province_options)

# Automatically determine postal code based on district, subdistrict, and province
postal_code = ""
if district and subdistrict and province:
    postal_code = postal_code_mapping.get((district, subdistrict, province), "")

st.text_input("รหัสไปรษณีย์ (Postal Code):", value=postal_code, disabled=True)

# Geocoder to fetch latitude and longitude for location names
geolocator = Nominatim(user_agent="ner-app")

def get_location_coordinates(location):
    """Fetch the latitude and longitude for a given location name."""
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return location_data.latitude, location_data.longitude
    except Exception as e:
        st.error(f"Error fetching location data: {e}")
    return None, None  # Return None if location could not be found

# Run button
if st.button("Run"):
    # Combine all inputs into a single text for processing
    input_text = f"{name} {address} {district} {subdistrict} {province} {postal_code}"

    # Run predictions on the combined input text
    results = predict_entities(model, input_text)

    # Display prediction results in a table
    st.subheader("Prediction Results")
    result_df = pd.DataFrame(results, columns=["Token", "Entity"])
    st.dataframe(result_df)

    # Visualization of predictions with color-coding
    st.subheader("Entity Visualization")
    locs = []  # List to store coordinates for locations

    for token, entity in results:
        color = "#FFCCCB" if entity == "LOC" else "#D3D3D3"
        st.markdown(f"<span style='background-color:{color}'>{token} ({entity})</span>", unsafe_allow_html=True)

        # If the entity is a location (LOC), get its latitude and longitude
        if entity == "LOC":
            lat, lon = get_location_coordinates(token)
            if lat and lon:
                locs.append((lat, lon))

    # Display the map if there are locations
    if locs:
        st.subheader("Location Map")
        # Convert location data to a DataFrame for use with st.map()
        location_df = pd.DataFrame(locs, columns=["latitude", "longitude"])
        st.map(location_df)
    else:
        st.warning("No location entities detected.")
