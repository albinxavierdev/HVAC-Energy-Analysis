# 🌀 Dynamic Model Prediction Control System for HVAC

## Overview

This project implements a Dynamic Model Prediction Control System for
HVAC (Heating, Ventilation, and Air Conditioning) systems. HVAC systems
are critical for maintaining indoor comfort and energy efficiency in
buildings. Traditionally, they operate on fixed schedules or static
rules, which often leads to wasted energy and inconsistent performance.

Our approach integrates neural networks and live sensor data to
dynamically adjust control strategies for equipment such as Air Handling
Units (AHUs). By predicting equipment behavior and optimizing control
decisions in real time, the system enables energy savings, improved
efficiency, and better occupant comfort.

## ✨ Key Features

-   **Dynamic Rules & Control**: Automatically adjusts HVAC operations
    based on real-time conditions and predicted performance.
-   **Live Sensor Integration**: Uses BACnet and InfluxDB data streams
    to continuously monitor HVAC equipment and building environment.
-   **Neural Network Predictions**: A feed-forward neural network (FFNN)
    is used to predict outcomes (e.g., temperature, airflow, energy
    consumption).
-   **Adaptive Optimization**: The system learns patterns from
    historical and live data to improve efficiency over time.
-   **Scalable Architecture**: Built on PostgreSQL for rules/control
    metadata and InfluxDB for high-frequency sensor data.
