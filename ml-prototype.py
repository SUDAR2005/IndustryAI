import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import bcrypt
import datetime
from plotly import graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import pipeline
from scipy.optimize import minimize
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# MongoDB setup
load_dotenv()
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client['greeninvestments']

class Authentication:
    def __init__(self):
        self.users = db.users
        
    def hash_password(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    def verify_password(self, password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    def signup(self, username, password, email):
        if self.users.find_one({"username": username}):
            return False, "Username already exists"
        
        user_data = {
            "username": username,
            "password": self.hash_password(password),
            "email": email,
            "created_at": datetime.datetime.now()
        }
        self.users.insert_one(user_data)
        return True, "Signup successful"
    
    def login(self, username, password):
        user = self.users.find_one({"username": username})
        if user and self.verify_password(password, user['password']):
            return True, user
        return False, None

class MLManager:
    def __init__(self):
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Warning: Sentiment analyzer initialization failed: {e}")
            self.sentiment_analyzer = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.project_scorer = RandomForestRegressor(random_state=42)
        
    def preprocess_data(self, df):
        """Preprocess numerical features for ML models"""
        numerical_columns = [
            'total_investment_required', 'expected_roi', 'financial_risk_score',
            'carbon_reduction_tons', 'energy_efficiency_score', 'jobs_created',
            'community_benefit_score', 'environmental_score', 'social_score',
            'governance_score', 'overall_esg_score'
        ]
        
        # Handle missing values
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Ensure all required columns exist
        for col in numerical_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[numerical_columns]
    
    def analyze_project_description(self, description):
        """Use NLP to analyze project descriptions"""
        if self.sentiment_analyzer is None:
            return {'sentiment_score': 0, 'confidence': 0}
            
        try:
            sentiment = self.sentiment_analyzer(description)[0]
            sentiment_score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
            return {
                'sentiment_score': sentiment_score,
                'confidence': sentiment['score']
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'sentiment_score': 0, 'confidence': 0}
    
    def calculate_project_score(self, df):
        """Calculate ML-based project scores and feature importance"""
        try:
            X = self.preprocess_data(df)
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
            
            # If we have ESG scores, use them as target, otherwise create a composite score
            if 'overall_esg_score' in df.columns:
                y = df['overall_esg_score']
            else:
                y = (X_scaled_df['expected_roi'] * 0.3 + 
                     X_scaled_df['environmental_score'] * 0.3 +
                     X_scaled_df['social_score'] * 0.2 +
                     X_scaled_df['governance_score'] * 0.2)
            
            # Train the model
            self.project_scorer.fit(X_scaled, y)
            
            # Generate project scores
            project_scores = self.project_scorer.predict(X_scaled)
            
            # Calculate feature importance
            feature_importance = dict(zip(X.columns, self.project_scorer.feature_importances_))
            
            return project_scores, feature_importance
            
        except Exception as e:
            print(f"Error in calculate_project_score: {e}")
            default_scores = np.zeros(len(df))
            default_importance = {col: 0 for col in self.preprocess_data(df).columns}
            return default_scores, default_importance
    
    def detect_anomalies(self, df):
        """Detect anomalous projects using Isolation Forest"""
        try:
            X = self.preprocess_data(df)
            X_scaled = self.scaler.fit_transform(X)
            anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
            return anomaly_scores
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return np.ones(len(df))
    
    def optimize_portfolio(self, df, budget_constraint, risk_tolerance):
        """Optimize investment portfolio using ML insights"""
        try:
            # Ensure we have required columns
            required_columns = ['expected_roi', 'total_investment_required', 'overall_esg_score']
            if not all(col in df.columns for col in required_columns):
                print("Missing required columns for portfolio optimization")
                return np.array([1/len(df)] * len(df))
            
            n_projects = len(df)
            
            # Extract required arrays
            expected_roi = np.array(df['expected_roi'].values)
            total_investment = np.array(df['total_investment_required'].values)
            esg_scores = np.array(df['overall_esg_score'].values)
            
            # Calculate covariance matrix using only relevant financial metrics
            financial_columns = [
                'expected_roi',
                'financial_risk_score',
                'market_volatility_index',
                'currency_risk_exposure'
            ]
            
            # Filter for available financial columns
            available_financial_columns = [col for col in financial_columns if col in df.columns]
            if not available_financial_columns:
                # If no financial columns are available, use identity matrix
                correlations = np.eye(n_projects)
            else:
                # Calculate correlation matrix using available financial columns
                financial_data = df[available_financial_columns]
                correlations = financial_data.corr().fillna(0).values
                
                # If correlation matrix is not square or wrong size, use identity matrix
                if correlations.shape != (n_projects, n_projects):
                    correlations = np.eye(n_projects)
                    
            def objective(weights):
                weights = np.array(weights)
                try:
                    portfolio_return = np.sum(weights * expected_roi)
                    portfolio_risk = np.sqrt(weights.T @ correlations @ weights)
                    esg_impact = np.sum(weights * esg_scores)
                    return -(portfolio_return + esg_impact - risk_tolerance * portfolio_risk)
                except Exception as e:
                    print(f"Error in objective function: {e}")
                    return np.inf

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x * total_investment)}
            ]
            
            bounds = tuple((0, 1) for _ in range(n_projects))
            initial_weights = np.array([1/n_projects] * n_projects)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'ftol': 1e-8, 'maxiter': 1000}
            )
            
            if result.success:
                weights = np.maximum(result.x, 0)
                weights = weights / np.sum(weights)
                return weights
            else:
                print(f"Optimization failed: {result.message}")
                return initial_weights
                
        except Exception as e:
            print(f"Portfolio optimization failed: {e}")
            return np.array([1/len(df)] * len(df)) 
    
    def cluster_projects(self, df):
        """Cluster projects based on their characteristics"""
        try:
            X = self.preprocess_data(df)
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            
            n_clusters = min(5, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_pca)
            
            return clusters, X_pca
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return np.zeros(len(df)), np.zeros((len(df), 3))
        
class DataManager:
    def __init__(self):
        self.projects = db.projects
        self.ml_manager = MLManager()
    
    def get_user_projects(self, user_id):
        """
        Retrieve all projects for a specific user
        
        Parameters:
        user_id (str): The ID of the user whose projects to retrieve
        
        Returns:
        list: List of project dictionaries
        """
        try:
            # Convert MongoDB cursor to list and exclude MongoDB _id field
            projects = list(self.projects.find(
                {"user_id": user_id},
                {'_id': 0}  # Exclude MongoDB _id field
            ))
            
            # Handle case where no projects are found
            if not projects:
                return []
                
            return projects
            
        except Exception as e:
            print(f"Error retrieving user projects: {e}")
            return []
    
    def process_date_columns(self, df):
        """Convert date strings to datetime objects"""
        date_columns = ['start_date', 'expected_completion_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    
    def validate_csv_columns(self, df):
        """Validate that the CSV has the required columns"""
        required_columns = [
            'project_id', 'project_name', 'project_description', 'project_type',
            'project_status', 'start_date', 'expected_completion_date', 'country',
            'region', 'coordinates', 'total_investment_required', 'current_funding',
            'funding_gap', 'expected_roi', 'payback_period', 'capital_expenditure',
            'operational_expenditure', 'maintenance_costs', 'projected_revenue_5yr',
            'projected_cashflow_5yr', 'financial_risk_score', 'market_volatility_index',
            'currency_risk_exposure', 'carbon_reduction_tons', 'carbon_intensity',
            'emission_savings_forecast', 'water_usage_reduction', 'energy_efficiency_score',
            'renewable_energy_generation', 'biodiversity_impact_score', 'land_use_change',
            'waste_reduction_tons', 'climate_risk_score', 'natural_disaster_risk',
            'climate_adaptation_score', 'jobs_created', 'community_benefit_score',
            'local_business_impact', 'healthcare_impact_score', 'education_impact_score',
            'poverty_reduction_impact', 'community_engagement_level', 'indigenous_peoples_impact',
            'stakeholder_satisfaction_score', 'compliance_score', 'transparency_index',
            'corruption_risk_score', 'management_experience_score', 'track_record_score',
            'regulatory_compliance_score', 'reporting_quality_score', 'monitoring_framework_score',
            'audit_frequency', 'environmental_score', 'social_score', 'governance_score',
            'overall_esg_score'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        return len(missing_columns) == 0, missing_columns
    
    def insert_projects_from_csv(self, df, user_id):
        """Insert multiple projects from a DataFrame"""
        successful_inserts = 0
        failed_inserts = 0
        
        for _, row in df.iterrows():
            try:
                project_data = row.to_dict()
                project_data['user_id'] = user_id
                project_data['created_at'] = datetime.datetime.now()
                
                # Convert datetime objects to strings for MongoDB
                if isinstance(project_data.get('start_date'), pd.Timestamp):
                    project_data['start_date'] = project_data['start_date'].strftime('%Y-%m-%d')
                if isinstance(project_data.get('expected_completion_date'), pd.Timestamp):
                    project_data['expected_completion_date'] = project_data['expected_completion_date'].strftime('%Y-%m-%d')
                
                self.projects.insert_one(project_data)
                successful_inserts += 1
            except Exception as e:
                print(f"Error inserting project: {e}")
                failed_inserts += 1
        
        return successful_inserts, failed_inserts
    
    def calculate_dashboard_metrics(self, projects_df):
        """Enhanced dashboard metrics with ML insights"""
        if projects_df.empty:
            return None
        
        try:
            # Basic metrics
            basic_metrics = {
                'total_projects': len(projects_df),
                'total_investment': projects_df['total_investment_required'].sum(),
                'average_roi': projects_df['expected_roi'].mean(),
                'total_carbon_reduction': projects_df['carbon_reduction_tons'].sum(),
                'avg_esg_score': projects_df['overall_esg_score'].mean(),
                'jobs_created': projects_df['jobs_created'].sum()
            }
    
            ml_scores, feature_importance = self.ml_manager.calculate_project_score(projects_df)
            anomaly_scores = self.ml_manager.detect_anomalies(projects_df)
            anomaly_count = sum(anomaly_scores == -1)
            
            optimal_weights = self.ml_manager.optimize_portfolio(
                projects_df,
                budget_constraint=basic_metrics['total_investment'] * 1.2,  # 20% buffer
                risk_tolerance=0.5
            )
            
            clusters, pca_components = self.ml_manager.cluster_projects(projects_df)
            
            # Add ML metrics
            ml_metrics = {
                'avg_project_score': float(np.mean(ml_scores)),
                'ml_scores': ml_scores.tolist(),
                'feature_importance': feature_importance,
                'top_feature': max(feature_importance.items(), key=lambda x: x[1])[0],
                'anomaly_count': int(anomaly_count),
                'optimal_allocation': optimal_weights.tolist(),
                'cluster_assignments': clusters.tolist(),
                'pca_components': pca_components.tolist()
            }
            
            return {**basic_metrics, **ml_metrics}
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return basic metrics if ML metrics fail
            return {
                'total_projects': len(projects_df),
                'total_investment': projects_df['total_investment_required'].sum(),
                'average_roi': projects_df['expected_roi'].mean(),
                'total_carbon_reduction': projects_df['carbon_reduction_tons'].sum(),
                'avg_esg_score': projects_df['overall_esg_score'].mean(),
                'jobs_created': projects_df['jobs_created'].sum()
            }
def display_ml_dashboard(df, metrics):
        """Display ML-enhanced dashboard"""
        print("Hello All")
        try:
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"${metrics['total_investment']:,.2f}")
                if 'total_carbon_reduction' in metrics:
                    st.metric("Total Carbon Reduction", f"{metrics['total_carbon_reduction']:,.0f} tons")
            with col2:
                st.metric("Average ROI", f"{metrics['average_roi']:.1%}")
                if 'avg_esg_score' in metrics:
                    st.metric("Average ESG Score", f"{metrics['avg_esg_score']:.2f}")
            with col3:
                st.metric("Total Projects", metrics['total_projects'])
                if 'jobs_created' in metrics:
                    st.metric("Jobs Created", f"{metrics['jobs_created']:,.0f}")
            
            # Only show ML visualizations if ML metrics are available
            if 'ml_scores' in metrics:
                # Project clusters visualization
                st.subheader("Project Clustering Analysis")
                pca_df = pd.DataFrame(
                    metrics['pca_components'], 
                    columns=['PC1', 'PC2', 'PC3']
                )
                pca_df['Cluster'] = metrics['cluster_assignments']
                
                fig_clusters = px.scatter_3d(
                    pca_df, x='PC1', y='PC2', z='PC3',
                    color='Cluster', title="Project Clusters in 3D Space"
                )
                st.plotly_chart(fig_clusters)
                
                # Portfolio optimization visualization
                if 'optimal_allocation' in metrics:
                    st.subheader("Optimal Portfolio Allocation")
                    allocation_df = pd.DataFrame({
                        'Project': df['project_name'],
                        'Optimal Weight': metrics['optimal_allocation']
                    })
                    fig_allocation = px.bar(
                        allocation_df, x='Project', y='Optimal Weight',
                        title="Recommended Investment Distribution"
                    )
                    st.plotly_chart(fig_allocation)
                    
                # # Project scores distribution
                # st.subheader("ML-Based Project Scores")
                # fig_scores = px.histogram(
                #     pd.DataFrame({'ML Score': metrics['ml_scores']}), x='ML Score',
                #     title="Distribution of Project Scores"
                # )
                # st.plotly_chart(fig_scores)
                
                # Feature importance
                if 'feature_importance' in metrics:
                    st.subheader("Feature Importance in Project Evaluation")
                    importance_df = pd.DataFrame({
                        'Feature': list(metrics['feature_importance'].keys()),
                        'Importance': list(metrics['feature_importance'].values())
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df, x='Feature', y='Importance',
                        title="Feature Importance in Project Evaluation"
                    )
                    st.plotly_chart(fig_importance)
        
        except Exception as e:
            st.error(f"Error displaying dashboard: {e}")
            # Display basic metrics as fallback
            st.metric("Total Projects", metrics['total_projects'])
            st.metric("Total Investment", f"${metrics['total_investment']:,.2f}")
            st.metric("Average ROI", f"{metrics['average_roi']:.1%}")

def main():
    st.set_page_config(page_title="Green Finance Optimization Platform 💵💹", layout="wide")
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'auth'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    auth = Authentication()
    data_manager = DataManager()
    
    if st.session_state.page == 'auth':
        st.title("Green Finance Optimization Platform")
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            st.header("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                success, user = auth.login(login_username, login_password)
                if success:
                    st.session_state.page = 'home'
                    st.session_state.user_id = str(user['_id'])
                    st.session_state.username = user['username']
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            st.header("Signup")
            signup_username = st.text_input("Username", key="signup_username")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_email = st.text_input("Email", key="signup_email")
            if st.button("Signup"):
                success, message = auth.signup(signup_username, signup_password, signup_email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Home Page
    elif st.session_state.page == 'home':
        st.title(f"Welcome, {st.session_state.username}!")
        
        # Sidebar for navigation
        with st.sidebar:
            st.title("Navigation")
            if st.button("Dashboard"):
                st.session_state.page = 'dashboard'
                st.rerun()
            if st.button("Update Database"):
                st.session_state.page = 'update_data'
                st.rerun()
            if st.button("Logout"):
                st.session_state.page = 'auth'
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
        
        # Main content area
        st.header("Quick Overview")
        projects = data_manager.get_user_projects(st.session_state.user_id)
        if projects:
            df = pd.DataFrame(projects)
            metrics = data_manager.calculate_dashboard_metrics(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Projects", metrics['total_projects'])
            with col2:
                st.metric("Total Investment", f"${metrics['total_investment']:,.2f}")
            with col3:
                st.metric("Average ROI", f"{metrics['average_roi']:.1%}")
            
            # Quick visualization
            st.subheader("Project Types Distribution")
            fig = px.pie(df, names='project_type', title="Projects by Type")
            st.plotly_chart(fig)
            
        else:
            st.info("No projects found. Start by adding some projects in the Update Database section.")
    
    elif st.session_state.page == 'dashboard':
        st.title("Dashboard")
        
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            if st.button("Home"):
                st.session_state.page = 'home'
                st.rerun()
            if st.button("Update Database"):
                st.session_state.page = 'update_data'
                st.rerun()
            if st.button("Logout"):
                st.session_state.page = 'auth'
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
        
        # Dashboard content
        projects = data_manager.get_user_projects(st.session_state.user_id)
        
        if projects:
            df = pd.DataFrame(projects)
            metrics = data_manager.calculate_dashboard_metrics(df)
            
            # Call the ML dashboard display function
            display_ml_dashboard(df, metrics)
            # Additional visualizations (if needed)
            st.subheader("Investment by Project Type")
            fig1 = px.bar(df, x='project_type', y='total_investment_required',
                         title="Investment Distribution by Project Type")
            st.plotly_chart(fig1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ESG Scores Distribution")
                fig2 = px.scatter_3d(df, x='environmental_score', y='social_score',
                                   z='governance_score', color='overall_esg_score',
                                   size='total_investment_required')
                st.plotly_chart(fig2)
            
            with col2:
                st.subheader("Risk Analysis")
                fig3 = px.scatter(df, x='expected_roi', y='financial_risk_score',
                                size='total_investment_required', color='project_type',
                                title="Risk vs Return Analysis")
                st.plotly_chart(fig3)
            
            # Project Details Table
            st.subheader("Project Details")
            display_columns = ['project_name', 'project_type', 'total_investment_required',
                             'expected_roi', 'overall_esg_score', 'project_status']
            st.dataframe(df[display_columns])
            
        else:
            st.info("No projects found. Start by adding some projects in the Update Database section.")
            
    # Update Database Page
    elif st.session_state.page == 'update_data':
        st.title("Update Database")
        
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            if st.button("Home"):
                st.session_state.page = 'home'
                st.rerun()
            if st.button("Dashboard"):
                st.session_state.page = 'dashboard'
                st.rerun()
            if st.button("Logout"):
                st.session_state.page = 'auth'
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
        
        # CSV Upload Section
        st.header("Upload Projects Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                is_valid, missing_columns = data_manager.validate_csv_columns(df)
                
                if not is_valid:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.stop()
                
                # Process dates
                df = data_manager.process_date_columns(df)
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Upload button
                if st.button("Upload to Database"):
                    with st.spinner("Uploading projects..."):
                        successful, failed = data_manager.insert_projects_from_csv(df, st.session_state.user_id)
                        
                        if successful > 0:
                            st.success(f"Successfully uploaded {successful} projects")
                        if failed > 0:
                            st.error(f"Failed to upload {failed} projects")
                        
                        # Show total number of projects after upload
                        total_projects = len(data_manager.get_user_projects(st.session_state.user_id))
                        st.info(f"Total projects in database: {total_projects}")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Display column information
        with st.expander("CSV Format Information"):
            st.markdown("""
            ### Required Columns:
            Your CSV file should include the following columns:
            
            1. Basic Project Information:
               - project_id
               - project_name
               - project_description
               - project_type
               - project_status
               - start_date
               - expected_completion_date
            
            2. Location Information:
               - country
               - region
               - coordinates
            
            3. Financial Metrics:
               - total_investment_required
               - current_funding
               - funding_gap
               - expected_roi
               - payback_period
               - capital_expenditure
               - operational_expenditure
               - maintenance_costs
               - projected_revenue_5yr
               - projected_cashflow_5yr
            
            4. Risk Metrics:
               - financial_risk_score
               - market_volatility_index
               - currency_risk_exposure
            
            5. Environmental Metrics:
               - carbon_reduction_tons
               - carbon_intensity
               - emission_savings_forecast
               - water_usage_reduction
               - energy_efficiency_score
               - renewable_energy_generation
               - biodiversity_impact_score
               - land_use_change
               - waste_reduction_tons
            
            6. Climate Metrics:
               - climate_risk_score
               - natural_disaster_risk
               - climate_adaptation_score
            
            7. Social Impact Metrics:
               - jobs_created
               - community_benefit_score
               - local_business_impact
               - healthcare_impact_score
               - education_impact_score
               - poverty_reduction_impact
               - community_engagement_level
               - indigenous_peoples_impact
               - stakeholder_satisfaction_score
            
            8. Governance Metrics:
               - compliance_score
               - transparency_index
               - corruption_risk_score
               - management_experience_score
               - track_record_score
               - regulatory_compliance_score
               - reporting_quality_score
               - monitoring_framework_score
               - audit_frequency
            
            9. ESG Scores:
               - environmental_score
               - social_score
               - governance_score
               - overall_esg_score
            """)

if __name__ == "__main__":
    main()