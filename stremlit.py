import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd

# --- 1. MODEL ARCHITECTURE ---
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.model(x)

# --- 2. LOAD SAVED ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_dim = scaler.n_features_in_
    
    model = ANN(input_dim=input_dim)
    model.load_state_dict(torch.load("placement_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model or Scaler files not found! Make sure 'placement_model.pth' and 'scaler.pkl' are in this directory.")
    st.stop()

# --- 3. STREAMLIT INTERFACE ---
st.title("🎓 Student Placement Prediction Dashboard")
st.write("Predict placement eligibility based on student academic and residency performance.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=40, value=22, step=1)
    gender_input = st.selectbox("Gender", ["Female", "Male"])
    stream_input = st.selectbox("Stream", [
        "Civil", 
        "Computer Science", 
        "Electrical", 
        "Electronics And Communication", 
        "Information Technology", 
        "Mechanical"
    ])

with col2:
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1, step=1)
    
    # Hostel input component reinstated
    hostel_input = st.selectbox("Living in Hostel?", ["No", "Yes"])
    hostel = 1 if hostel_input == "Yes" else 0
    
    backlog_input = st.selectbox("History of Backlogs?", ["No", "Yes"])
    backlogs = 1 if backlog_input == "Yes" else 0

st.markdown("---")

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Placement Status", type="primary"):
    
    # Recreate the exact 11 training features matching your original index order:
    # ['Age', 'Internships', 'CGPA', 'Hostel', 'HistoryOfBacklogs', 'Stream_...', 'Gender_Male']
    features = {
        'Age': age,
        'Internships': internships,
        'CGPA': cgpa,
        'Hostel': hostel,
        'HistoryOfBacklogs': backlogs,
        # One-Hot Encoded Streams (Civil is dropped as the baseline first column)
        'Stream_Computer Science': 1 if stream_input == "Computer Science" else 0,
        'Stream_Electrical': 1 if stream_input == "Electrical" else 0,
        'Stream_Electronics And Communication': 1 if stream_input == "Electronics And Communication" else 0,
        'Stream_Information Technology': 1 if stream_input == "Information Technology" else 0,
        'Stream_Mechanical': 1 if stream_input == "Mechanical" else 0,
        # One-Hot Encoded Gender (Female is dropped as the baseline)
        'Gender_Male': 1 if gender_input == "Male" else 0
    }
    
    # Convert to DataFrame to match expected feature column structure for the scaler
    final_features_df = pd.DataFrame([features])
    
    # Scale inputs using your original 11-feature scaler
    scaled_features = scaler.transform(final_features_df)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    
    # Run the model forward pass
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    status_class = prediction.item()
    confidence_pct = confidence.item() * 100
    
    if status_class == 1:
        st.success(f"🎉 **Predicted Status: Placed!** (Confidence: {confidence_pct:.2f}%)")
        st.balloons()
    else:
        st.error(f"❌ **Predicted Status: Not Placed** (Confidence: {confidence_pct:.2f}%)")