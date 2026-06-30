import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# --- 1. RECREATE THE MODEL ARCHITECTURE ---
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
    # Load scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    # Determine input dimensions from scaler (tells us how many features it expects)
    input_dim = scaler.n_features_in_
    
    # Load model weights
    model = ANN(input_dim=input_dim)
    model.load_state_dict(torch.load("placement_model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model or Scaler files not found! Make sure 'placement_model.pth' and 'scaler.pkl' are in this directory.")
    st.stop()

# --- 3. STREAMLIT USER INTERFACE ---
st.title("🎓 Student Placement Prediction Dashboard")
st.write("Enter the student's metrics below to predict their placement status.")

st.markdown("---")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=40, value=22, step=1)
    
    # Matching the LabelEncoder mapping used during your training
    gender_input = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_input == "Male" else 0
    
    # Adjust choices based on your specific dataset labels if needed
    stream_input = st.selectbox("Stream", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Electrical"])
    # Dynamic stream encoding logic or a manual dictionary can go here depending on your encoder mapping.
    # For now, using a placeholder encoding (0, 1, 2, ...). Match this to your exact label encoder.
    stream_mapping = {"Civil": 0, "Computer Science": 1, "Electrical": 2, "Electronics": 3, "Information Technology": 4, "Mechanical": 5}
    stream = stream_mapping.get(stream_input, 0)

    internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1, step=1)

with col2:
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    
    hostel_input = st.selectbox("Living in Hostel?", ["No", "Yes"])
    hostel = 1 if hostel_input == "Yes" else 0
    
    backlog_input = st.selectbox("History of Backlogs?", ["No", "Yes"])
    backlogs = 1 if backlog_input == "Yes" else 0

st.markdown("---")

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Placement Status", type="primary"):
    # Arrange raw features in the exact column order your model trained on:
    # [Age, Gender, Stream, Internships, CGPA, Hostel, HistoryOfBacklogs]
    raw_features = np.array([[age, gender, stream, internships, cgpa, hostel, backlogs]], dtype=np.float32)
    
    # Scale inputs using the loaded scaler
    scaled_features = scaler.transform(raw_features)
    
    # Convert to PyTorch Tensor
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    # Display Results
    status_class = prediction.item()
    confidence_pct = confidence.item() * 100
    
    if status_class == 1:
        st.success(f"🎉 **Predicted Status: Placed!** (Confidence: {confidence_pct:.2f}%)")
        st.balloons()
    else:
        st.sidebar.warning("⚠️ High Risk Candidate")
        st.error(f"❌ **Predicted Status: Not Placed** (Confidence: {confidence_pct:.2f}%)")