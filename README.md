# Cyber_Threat_Detection
Machine learning system that detects cyber threats using only one-way (unidirectional) network traffic, as seen on data-diode gateways. Trains Random Forest, SVM and Gradient Boosting on CICIDS2017-style flow data, comparing forward-only vs full bidirectional features to show detection stays accurate (~98.4% F1) without return-path visibility.
