import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from pymongo import MongoClient
import requests
import json
from typing import Dict, List, Union
import logging

class DataPipeline:
    def __init__(self, config: Dict):
        """
        Initialize data pipeline with configuration
        
        config: Dictionary containing:
            - database credentials
            - API endpoints
            - file paths
            - data schemas
        """
        self.config = config
        self.sql_engine = create_engine(config['sql_connection_string'])
        self.mongo_client = MongoClient(config['mongo_connection_string'])
        self.logger = logging.getLogger(__name__)
        
    def ingest_structured_data(self) -> pd.DataFrame:
        
        financial_data = self._fetch_financial_metrics()
        
        esg_data = self._fetch_esg_metrics()
        
        climate_data = self._fetch_climate_data()
        
        merged_data = pd.merge(financial_data, esg_data, on='project_id')
        merged_data = pd.merge(merged_data, climate_data, on='project_id')
        
        return merged_data
    
    def ingest_unstructured_data(self) -> Dict:
        """Ingest unstructured data like documents and reports"""
        
        project_docs = self._fetch_project_documents()
        
        sustainability_reports = self._fetch_sustainability_reports()
        
        news_data = self._fetch_news_data()
        
        return {
            'project_documents': project_docs,
            'sustainability_reports': sustainability_reports,
            'news_data': news_data
        }
    
    def _fetch_financial_metrics(self) -> pd.DataFrame:
        """Fetch financial data from various sources"""
        dfs = []
        
        query = """
        SELECT project_id, investment_amount, expected_roi, 
               implementation_cost, maintenance_cost
        FROM financial_metrics
        WHERE data_date >= NOW() - INTERVAL '1 year'
        """
        internal_data = pd.read_sql(query, self.sql_engine)
        dfs.append(internal_data)
        
        for api_config in self.config['financial_apis']:
            response = requests.get(
                api_config['endpoint'],
                headers={'Authorization': api_config['key']}
            )
            if response.status_code == 200:
                api_data = pd.DataFrame(response.json())
                dfs.append(api_data)
        
        return pd.concat(dfs, ignore_index=True)
    
    def _fetch_esg_metrics(self) -> pd.DataFrame:
        """Fetch ESG data from various sources"""
        
        esg_collection = self.mongo_client.green_finance.esg_metrics
        
        esg_data = []
        
        emissions_data = pd.DataFrame(list(esg_collection.find(
            {'metric_type': 'emissions'}
        )))
        
        social_data = pd.DataFrame(list(esg_collection.find(
            {'metric_type': 'social_impact'}
        )))
        
        governance_data = pd.DataFrame(list(esg_collection.find(
            {'metric_type': 'governance'}
        )))
        
        return pd.concat([emissions_data, social_data, governance_data], 
                        ignore_index=True)
    
    def _fetch_climate_data(self) -> pd.DataFrame:
        
        climate_data = []
        
        for weather_api in self.config['weather_apis']:
            response = requests.get(weather_api['endpoint'])
            if response.status_code == 200:
                climate_data.append(pd.DataFrame(response.json()))
        
        env_impact_query = """
        SELECT project_id, impact_type, impact_score
        FROM environmental_impacts
        WHERE assessment_date >= NOW() - INTERVAL '6 months'
        """
        env_impacts = pd.read_sql(env_impact_query, self.sql_engine)
        climate_data.append(env_impacts)
        
        return pd.concat(climate_data, ignore_index=True)
    
    def preprocess_data(self, structured_data: pd.DataFrame, 
                       unstructured_data: Dict) -> Dict:
        """Preprocess and clean the data"""
        structured_data = self._handle_missing_values(structured_data)
        
        structured_data = self._engineer_features(structured_data)
        
        processed_docs = self._preprocess_documents(unstructured_data)
        
        structured_data = self._normalize_features(structured_data)
        
        return {
            'structured_data': structured_data,
            'processed_documents': processed_docs
        }
    
    def validate_data(self, data: Dict) -> bool:
        """Validate data quality and consistency"""
        
        validation_results = []
        
        required_fields = self.config['required_fields']
        fields_present = all(field in data['structured_data'].columns 
                           for field in required_fields)
        validation_results.append(fields_present)
        
        correct_types = self._validate_data_types(data['structured_data'])
        validation_results.append(correct_types)
        
        valid_ranges = self._validate_value_ranges(data['structured_data'])
        validation_results.append(valid_ranges)
        
        return all(validation_results)
    
    
class DataLoader:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
    
    def load_data(self) -> Dict:
        """Main method to load and prepare all data"""
        
        structured_data = self.pipeline.ingest_structured_data()
        unstructured_data = self.pipeline.ingest_unstructured_data()
        
        processed_data = self.pipeline.preprocess_data(
            structured_data, unstructured_data
        )
        
        if not self.pipeline.validate_data(processed_data):
            raise ValueError("Data validation failed")
        
        return processed_data