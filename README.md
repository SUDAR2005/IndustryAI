# Green Finance Optimization Platform

## Overview
The Green Finance Optimization Platform is a machine learning-powered web application designed to help users evaluate, analyze, and optimize sustainable investment projects. The platform provides advanced tools for data analysis, project scoring, anomaly detection, portfolio optimization, and ESG (Environmental, Social, and Governance) insights.

## Features
- **User Authentication**: Secure signup and login system using hashed passwords.
- **Project Dashboard**: Visualize key project metrics, such as total investment, ROI, carbon reduction, and ESG scores.
- **ML Insights**:
  - Sentiment analysis of project descriptions.
  - Project clustering based on PCA and K-Means.
  - Anomaly detection using Isolation Forest.
  - Feature importance analysis.
  - Portfolio optimization for investments.
- **Data Upload**: Import projects via CSV with extensive data validation.
- **Interactive Visualizations**: 3D scatter plots, bar charts, and more using Plotly.

## Technologies Used
- **Frontend**: Streamlit for the user interface.
- **Backend**: MongoDB for data storage.
- **Machine Learning**:
  - Scikit-learn for preprocessing, anomaly detection, and clustering.
  - Transformers for sentiment analysis.
  - TensorFlow for future extensibility.
- **Visualization**: Plotly for interactive dashboards.

## Installation

### Prerequisites
- Python 3.8+
- MongoDB server

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   #On Mac, use  source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the `.env` file with the MongoDB connection string:
   ```
   MONGO_URI=mongodb://localhost:27017
   ```
5. Run the application:
   ```bash
   streamlit run ml-prototype.py
   ```

## Usage
1. **User Authentication**: Sign up or log in to access the platform.
2. **Project Analysis**:
   - Navigate to the "Dashboard" to view project metrics.
   - Use clustering and feature importance tools for in-depth analysis.
3. **Data Upload**:
   - Go to "Update Database" to upload project data via CSV.
   - Ensure the file matches the required format (see CSV format information in the app).
4. **Portfolio Optimization**: Optimize investment allocation with machine learning insights.

## File Structure
- `ml-prototype.py`: Main application script.
- `requirements.txt`: Python dependencies.

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch and submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For questions or support, please contact [your-email@example.com].

