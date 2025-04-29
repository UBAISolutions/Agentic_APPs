
import pandas as pd
import numpy as np
# Set matplotlib backend to Agg (non-interactive, thread-safe)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from io import BytesIO
import base64
import re
from typing import Dict, Any, List, Union

class AnalyticsEngine:
    def __init__(self):
        pass
    
    def analyze_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], query: str) -> Dict[str, Any]:
        """Analyze data based on the query."""
        result = {
            "query": query,
            "response_type": "text",
            "content": "Analysis could not be performed."
        }
        
        # Convert dict of DataFrames to a single DataFrame if needed
        if isinstance(data, dict):
            if len(data) == 1:
                data = list(data.values())[0]
            else:
                # Set default to first sheet for now
                data = list(data.values())[0]
                
        # Check if we're dealing with a DataFrame
        if not isinstance(data, pd.DataFrame):
            return result
            
        # Determine the type of analysis requested
        if "chart" in query.lower() or "plot" in query.lower() or "graph" in query.lower():
            return self._generate_visualization(data, query)
        elif "forecast" in query.lower() or "predict" in query.lower():
            return self._generate_forecast(data, query)
        elif "excel" in query.lower() or "xlsx" in query.lower() or "export" in query.lower():
            return self._prepare_export(data, query)
        elif "trend" in query.lower() or "pattern" in query.lower():
            return self._analyze_trends(data, query)
        else:
            return self._general_analysis(data, query)
    
    def _generate_visualization(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate a visualization based on the query."""
        # Determine chart type based on query
        chart_type = "line"  # Default chart type
        if "bar" in query.lower() or "histogram" in query.lower():
            chart_type = "bar"
        elif "scatter" in query.lower():
            chart_type = "scatter"
        elif "pie" in query.lower():
            chart_type = "pie"
        
        # Set up the figure
        plt.figure(figsize=(12, 6))
        
        # Try to infer what columns to use based on the query
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "response_type": "text",
                "content": "No numeric columns found for visualization."
            }
            
        # Simple heuristic for column selection
        x_col = None
        y_col = None
        
        # Try to find date columns for time series
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]' 
                    or ('date' in col.lower() and pd.to_datetime(data[col], errors='coerce').notna().all())]
        
        if date_cols:
            x_col = date_cols[0]
            # Convert to datetime if not already
            data[x_col] = pd.to_datetime(data[x_col])
        else:
            # Use first column as x if not numeric
            non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
            if non_numeric_cols:
                x_col = non_numeric_cols[0]
        
        # For y, use the first numeric column
        if numeric_cols:
            y_col = numeric_cols[0]
            
        # Check if specific columns are mentioned in the query
        for col in data.columns:
            col_terms = [col.lower(), col.lower().replace("_", " "), col.lower().replace("%", "percent")]
            if any(term in query.lower() for term in col_terms):
                if col in numeric_cols:
                    y_col = col
                else:
                    x_col = col
        
        # Create the plot based on requested chart type
        if x_col and y_col:
            if chart_type == "bar":
                # Bar chart - always use this if requested
                if x_col in date_cols and data[x_col].nunique() > 15:
                    # If we have many date values, group by month/week
                    data['month'] = data[x_col].dt.to_period('M')
                    grouped = data.groupby('month')[y_col].mean().reset_index()
                    grouped['month'] = grouped['month'].astype(str)
                    sns.barplot(x='month', y=y_col, data=grouped)
                    plt.xlabel("Month")
                    plt.xticks(rotation=45)
                else:
                    sns.barplot(x=x_col, y=y_col, data=data)
                    plt.xticks(rotation=45)
            elif chart_type == "pie" and data[y_col].sum() > 0:
                # Pie chart
                if x_col:
                    plt.pie(data[y_col], labels=data[x_col], autopct='%1.1f%%')
                    plt.axis('equal')
                else:
                    return {
                        "response_type": "text",
                        "content": "Cannot create pie chart without category column."
                    }
            elif chart_type == "scatter":
                # Scatter plot
                sns.scatterplot(x=x_col, y=y_col, data=data)
            else:
                # Line chart - default
                if x_col in date_cols:
                    sns.lineplot(x=x_col, y=y_col, data=data)
                    plt.xticks(rotation=45)
                elif data[x_col].nunique() < 10:
                    # Categorical x with few values
                    sns.barplot(x=x_col, y=y_col, data=data)
                    plt.xticks(rotation=45)
                else:
                    # Many x values, use scatter
                    sns.scatterplot(x=x_col, y=y_col, data=data)
            
            # Set labels
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return {
                "response_type": "chart",
                "content": img_str,
                "format": "png",
                "encoding": "base64",
                "chart_type": chart_type,
                "description": f"{chart_type.capitalize()} chart showing {y_col.replace('_', ' ')} by {x_col.replace('_', ' ')}",
                "columns_used": [x_col, y_col]
            }
        else:
            return {
                "response_type": "text",
                "content": "Could not determine appropriate columns for visualization."
            }
    
    def _generate_forecast(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate a forecast using Prophet."""
        # Look for datetime column
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]' 
                    or ('date' in col.lower() and pd.to_datetime(data[col], errors='coerce').notna().all())]
        
        if not date_cols:
            return {
                "response_type": "text",
                "content": "No date column found for forecasting."
            }
        
        # Find a numeric column for the target
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "response_type": "text",
                "content": "No numeric columns found for forecasting."
            }
            
        # Prepare data for Prophet
        ds_col = date_cols[0]
        y_col = numeric_cols[0]
        
        # Check if specific columns are mentioned in the query
        for col in data.columns:
            col_terms = [col.lower(), col.lower().replace("_", " "), col.lower().replace("%", "percent")]
            if any(term in query.lower() for term in col_terms) and col in numeric_cols:
                y_col = col
        
        # Create dataframe for Prophet
        prophet_data = data[[ds_col, y_col]].copy()
        
        # Remove rows with NaN values
        prophet_data = prophet_data.dropna()
        
        # Sort by date
        prophet_data = prophet_data.sort_values(ds_col)
        
        # Check if we have enough data
        if len(prophet_data) < 5:
            return {
                "response_type": "text",
                "content": f"Not enough data points ({len(prophet_data)}) for forecasting. Need at least 5 data points."
            }
        
        # Rename columns for Prophet
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # Fit Prophet model
        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            m.fit(prophet_data)
        except Exception as e:
            return {
                "response_type": "text",
                "content": f"Error fitting forecast model: {str(e)}"
            }
        
        # Make future dataframe
        future_periods = 10  # Default to 10 days
        
        # Try to extract the forecast period from the query
        period_match = re.search(r'(\d+)\s*(day|week|month|year)s?', query.lower())
        if period_match:
            num = int(period_match.group(1))
            unit = period_match.group(2)
            
            if unit == 'day':
                future_periods = num
            elif unit == 'week':
                future_periods = num * 7
            elif unit == 'month':
                future_periods = num * 30
            elif unit == 'year':
                future_periods = num * 365
        
        # Determine frequency based on data
        time_diff = (prophet_data['ds'].max() - prophet_data['ds'].min()).total_seconds()
        avg_interval = time_diff / (len(prophet_data) - 1) if len(prophet_data) > 1 else 86400
        
        if avg_interval < 3600:  # Less than an hour
            freq = 'H'
        elif avg_interval < 86400:  # Less than a day
            freq = 'D'
        elif avg_interval < 604800:  # Less than a week
            freq = 'W'
        elif avg_interval < 2592000:  # Less than a month
            freq = 'M'
        else:  # More than a month
            freq = 'M'
        
        future = m.make_future_dataframe(periods=future_periods, freq=freq)
        forecast = m.predict(future)
        
        # Create plot
        fig = plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.plot(prophet_data['ds'], prophet_data['y'], 'b-', label='Historical Data')
        
        # Plot forecast
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods)
        plt.plot(forecast_data['ds'], forecast_data['yhat'], 'r-', label='Forecast')
        plt.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], 
                        forecast_data['yhat_upper'], color='r', alpha=0.2, label='Confidence Interval')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.title(f'Forecast of {y_col.replace("_", " ").title()} for next {future_periods} periods')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Prepare forecast data for output
        forecast_output = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_output['ds'] = forecast_output['ds'].dt.strftime('%Y-%m-%d')
        
        return {
            "response_type": "forecast",
            "chart": {
                "content": img_str,
                "format": "png",
                "encoding": "base64",
                "description": f"Forecast of {y_col.replace('_', ' ')} for next {future_periods} periods"
            },
            "data": forecast_output.to_dict('records'),
            "columns_used": [ds_col, y_col],
            "periods": future_periods
        }
    
    def _prepare_export(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Prepare data export based on the query."""
        # Create an in-memory Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=False)
        
        output.seek(0)
        excel_data = base64.b64encode(output.read()).decode()
        
        return {
            "response_type": "export",
            "content": excel_data,
            "format": "xlsx",
            "encoding": "base64",
            "filename": "data_export.xlsx",
            "description": "Exported data as requested",
            "rows": len(data),
            "columns": len(data.columns)
        }
    
    def _analyze_trends(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze trends in the data."""
        # Look for datetime column
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]' 
                    or ('date' in col.lower() and pd.to_datetime(data[col], errors='coerce').notna().all())]
        
        if not date_cols:
            return {
                "response_type": "text",
                "content": "No date column found for trend analysis."
            }
        
        # Find numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "response_type": "text",
                "content": "No numeric columns found for trend analysis."
            }
            
        ds_col = date_cols[0]
        
        # Convert to datetime if not already
        data[ds_col] = pd.to_datetime(data[ds_col])
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Plot trends for up to 5 numeric columns
        for i, y_col in enumerate(numeric_cols[:5]):
            plt.subplot(min(len(numeric_cols[:5]), 5), 1, i+1)
            sns.lineplot(x=ds_col, y=y_col, data=data)
            plt.title(f"Trend of {y_col} over time")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Calculate trend statistics
        trend_stats = {}
        for col in numeric_cols[:5]:
            trend_data = data[[ds_col, col]].dropna().sort_values(by=ds_col)
            
            if len(trend_data) > 1:
                # Calculate basic trend metrics
                first_value = trend_data[col].iloc[0]
                last_value = trend_data[col].iloc[-1]
                change = last_value - first_value
                pct_change = (change / first_value * 100) if first_value != 0 else float('inf')
                
                trend_stats[col] = {
                    "first_value": first_value,
                    "last_value": last_value,
                    "change": change,
                    "pct_change": pct_change
                }
        
        return {
            "response_type": "trend",
            "chart": {
                "content": img_str,
                "format": "png",
                "encoding": "base64",
                "description": "Trend analysis charts"
            },
            "trends": trend_stats,
            "time_column": ds_col,
            "analyzed_columns": numeric_cols[:5]
        }
    
    def _general_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform general analysis on the data."""
        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        stats = {}
        
        if numeric_cols:
            stats["basic"] = data[numeric_cols].describe().to_dict()
        
        # Missing values
        missing = data.isnull().sum().to_dict()
        
        # Data types
        dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Sample data
        sample = data.head(5).to_dict('records')
        
        return {
            "response_type": "analysis",
            "statistics": stats,
            "missing_values": missing,
            "data_types": dtypes,
            "sample_data": sample,
            "rows": len(data),
            "columns": len(data.columns)
        }
