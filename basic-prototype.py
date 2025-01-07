import streamlit as st
import pandas as pd
from pymongo import MongoClient
import bcrypt
import datetime
from plotly import graph_objects as go
import plotly.express as px
from dotenv import load_dotenv


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

class DataManager:
    def __init__(self):
        self.projects = db.projects
        
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
    
    def get_user_projects(self, user_id):
        return list(self.projects.find({"user_id": user_id}, {'_id': 0}))
    
    def calculate_dashboard_metrics(self, projects_df):
        """Calculate summary metrics for dashboard with graceful handling of missing columns"""
        if projects_df.empty:
            return None
        
        metrics = {
            'total_projects': len(projects_df),
            'total_investment': projects_df['total_investment_required'].sum() if 'total_investment_required' in projects_df.columns else 0,
            'average_roi': projects_df['expected_roi'].mean() if 'expected_roi' in projects_df.columns else 0,
            'total_carbon_reduction': projects_df['carbon_reduction_tons'].sum() if 'carbon_reduction_tons' in projects_df.columns else 0,
            'avg_esg_score': projects_df['overall_esg_score'].mean() if 'overall_esg_score' in projects_df.columns else 0,
            'jobs_created': projects_df['jobs_created'].sum() if 'jobs_created' in projects_df.columns else 0
        }
        
        # Format percentages and round numbers
        metrics['average_roi'] = metrics['average_roi'] if pd.isna(metrics['average_roi']) else float(metrics['average_roi'])
        metrics['avg_esg_score'] = round(float(metrics['avg_esg_score']), 2) if not pd.isna(metrics['avg_esg_score']) else 0
        
        return metrics

def main():
    st.set_page_config(page_title="Green Finance Platform", layout="wide")
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'auth'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    auth = Authentication()
    data_manager = DataManager()
    
    # Authentication Page
    if st.session_state.page == 'auth':
        st.title("Green Finance Platform")
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
    
    # Dashboard Page
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
            
            # Metrics Overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Investment", f"${metrics['total_investment']:,.2f}")
                st.metric("Total Carbon Reduction", f"{metrics['total_carbon_reduction']:,.0f} tons")
            with col2:
                st.metric("Average ROI", f"{metrics['average_roi']:.1%}")
                st.metric("Average ESG Score", f"{metrics['avg_esg_score']:.2f}")
            with col3:
                st.metric("Total Projects", metrics['total_projects'])
                st.metric("Jobs Created", f"{metrics['jobs_created']:,.0f}")
            
            # Visualizations
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
            
            . Financial Metrics:
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