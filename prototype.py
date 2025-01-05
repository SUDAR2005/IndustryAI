import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pulp import *
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class GreenFinancePrototype:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100)
        
    def preprocess_data(self, df):
        """Preprocess and clean the dataset"""
        # Convert dates to datetime
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['expected_completion_date'] = pd.to_datetime(df['expected_completion_date'])
        
        # Calculate project duration
        df['project_duration'] = (df['expected_completion_date'] - df['start_date']).dt.days
        
        # Create risk composite score
        df['risk_score'] = df[['financial_risk_score', 'climate_risk_score', 
                              'natural_disaster_risk', 'corruption_risk_score']].mean(axis=1)
        
        return df
    
    def calculate_esg_scores(self, df):
        """Calculate ESG scores using weighted averages of relevant metrics"""
        # Environmental score components
        env_metrics = ['carbon_reduction_tons', 'energy_efficiency_score', 
                      'renewable_energy_generation', 'biodiversity_impact_score']
        
        # Social score components
        social_metrics = ['jobs_created', 'community_benefit_score', 
                         'healthcare_impact_score', 'education_impact_score']
        
        # Governance score components
        gov_metrics = ['compliance_score', 'transparency_index', 
                      'regulatory_compliance_score', 'reporting_quality_score']
        
        # Normalize metrics
        df_scaled = pd.DataFrame()
        for metrics in [env_metrics, social_metrics, gov_metrics]:
            df_scaled[metrics] = self.scaler.fit_transform(df[metrics])
        
        # Calculate weighted scores
        df['environmental_score'] = df_scaled[env_metrics].mean(axis=1)
        df['social_score'] = df_scaled[social_metrics].mean(axis=1)
        df['governance_score'] = df_scaled[gov_metrics].mean(axis=1)
        df['overall_esg_score'] = (df['environmental_score'] + 
                                 df['social_score'] + 
                                 df['governance_score']) / 3
        
        return df
    
    def optimize_portfolio(self, df, total_budget, min_roi=0.05, max_risk=0.7):
        """Optimize project portfolio using linear programming"""
        # Create optimization problem
        prob = LpProblem("Green_Finance_Optimization", LpMaximize)
        
        # Decision variables
        project_vars = LpVariable.dicts("Project",
                                      ((i) for i in df.index),
                                      0, 1, LpBinary)
        
        # Objective function: Maximize ESG score
        prob += lpSum([project_vars[i] * df.loc[i, 'overall_esg_score'] 
                      for i in df.index])
        
        # Constraints
        # Budget constraint
        prob += lpSum([project_vars[i] * df.loc[i, 'total_investment_required'] 
                      for i in df.index]) <= total_budget
        
        # Minimum ROI constraint
        prob += lpSum([project_vars[i] * df.loc[i, 'expected_roi'] 
                      for i in df.index]) >= min_roi * total_budget
        
        # Maximum risk constraint
        prob += lpSum([project_vars[i] * df.loc[i, 'risk_score'] 
                      for i in df.index]) <= max_risk * len(df.index)
        
        # Solve the problem
        prob.solve()
        
        # Get selected projects
        selected_projects = [i for i in df.index if project_vars[i].value() == 1]
        return selected_projects
    
    def create_dashboard(self, df, selected_projects):
        """Create Streamlit dashboard with visualizations"""
        st.title("Green Finance Portfolio Optimization Dashboard")
        
        # Portfolio Overview
        st.header("Portfolio Overview")
        selected_df = df.loc[selected_projects]
        
        # ESG Score Distribution
        fig_esg = px.scatter_3d(selected_df, 
                               x='environmental_score',
                               y='social_score', 
                               z='governance_score',
                               color='overall_esg_score',
                               size='total_investment_required',
                               hover_data=['project_name'])
        st.plotly_chart(fig_esg)
        
        # Financial Metrics
        st.header("Financial Metrics")
        fig_fin = go.Figure(data=[
            go.Bar(name='Investment Required', 
                  x=selected_df['project_name'], 
                  y=selected_df['total_investment_required']),
            go.Bar(name='Expected ROI', 
                  x=selected_df['project_name'], 
                  y=selected_df['expected_roi'])
        ])
        st.plotly_chart(fig_fin)
        
        # Project Details
        st.header("Selected Projects")
        st.dataframe(selected_df[['project_name', 'project_type', 'overall_esg_score', 
                                'total_investment_required', 'expected_roi']])

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('sample_green_finance_data.csv')
        
    # Initialize the prototype
    gfp = GreenFinancePrototype()
        
    # Process data
    df_processed = gfp.preprocess_data(df)
    df_scored = gfp.calculate_esg_scores(df_processed)
        
    # Optimize portfolio
    selected_projects = gfp.optimize_portfolio(df_scored, 
                                                total_budget=1000000, 
                                                min_roi=0.05, 
                                                max_risk=0.7)
        
    # Create dashboard
    gfp.create_dashboard(df_scored, selected_projects)