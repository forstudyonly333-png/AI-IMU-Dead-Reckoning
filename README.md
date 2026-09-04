# 🛰️ AI-IMU Dead Reckoning

## GPS-Denied Position Estimation using Physics, EKF and LSTM

AI-IMU Dead Reckoning is a hybrid navigation prototype designed to estimate vehicle position when GPS becomes unavailable.

The system combines:

- Physics-based Dead Reckoning
- Extended Kalman Filter (EKF)
- LSTM-based AI correction
- Hybrid GPS + EKF + AI fusion
- GPS outage simulation
- Offline city-to-city navigation
- Road-based route visualization
- Interactive Streamlit dashboard

---

# 🚀 Problem Statement

GPS can become unavailable in environments such as:

- Tunnels
- Underground areas
- Indoor environments
- Dense urban environments
- GPS-denied zones

Traditional GPS navigation cannot continuously provide accurate positioning in these situations.

This project demonstrates a hybrid approach where IMU sensor data is used with physics-based navigation, EKF filtering and an LSTM neural network to estimate position during GPS outages.

---

# 🧠 Proposed Solution

The system follows this pipeline:

IMU Sensor Data
        ↓
Data Preprocessing
        ↓
Physics Dead Reckoning
        ↓
Extended Kalman Filter
        ↓
LSTM AI Correction
        ↓
Hybrid GPS + EKF + AI Fusion
        ↓
Position Estimation
        ↓
Navigation Dashboard

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Machine Learning

- PyTorch
- LSTM Neural Network
- NumPy

## Data Processing

- Pandas
- NumPy

## Navigation

- Physics-based Dead Reckoning
- Extended Kalman Filter
- IMU sensor processing
- GPS outage simulation
- Road routing

## Visualization

- Matplotlib
- Folium

## Dashboard

- Streamlit
- Streamlit-Folium

---

# 📂 Project Structure

```text
AI_IMU_DR_PROJECT/
│
├── data/
│
├── models/
│   └── ai_adapter.pth
│
├── results/
│   ├── final_results.csv
│   ├── trajectory.png
│   ├── trajectory_comparison.png
│   ├── error_comparison.png
│   ├── gps_outage.png
│   └── error_over_time.png
│
├── src/
│   ├── navigation.py
│   ├── hybrid_dr.py
│   ├── visualization.py
│   └── ...
│
├── dashboard/
│   └── app.py
│
├── train.py
├── test.py
├── requirements.txt
└── README.md"# LOGIC_LOOPERS" 
