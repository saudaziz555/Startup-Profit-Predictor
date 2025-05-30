# liabaries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Function to format currency values
def format_currency(value):
    return f"SAR {value:,.0f}"

# Set a consistent theme for all plotly charts
chart_theme = "plotly_dark"
color_palette = px.colors.qualitative.Bold
chart_bg_color = "rgba(36, 37, 45, 0.1)"

#loading model
model = joblib.load("model.pkl")

#header title
st.set_page_config(page_title="Predicting startup profits", layout="wide", 
                   initial_sidebar_state="expanded")

#sidebar
st.sidebar.markdown("<h1 style='font-size: 32px;'>Dashboard</h1>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload file", type=["xlsx"])

fil1 = st.sidebar.selectbox("Numerical filter", [None, 'R&D Spend', 'Administration', 'Marketing Spend'])
fil2 = st.sidebar.selectbox("Categorical filter", [None, 'State'])


st.sidebar.title("Predicted Companies Profit")
st.sidebar.markdown("Enter budget data to get forecast :")
rd = st.sidebar.number_input("R&D Spend", min_value=0.0, value=500000.0, format="%.2f")
admin = st.sidebar.number_input("Administration", min_value=0.0, value=500000.0, format="%.2f")
marketing = st.sidebar.number_input("Marketing Spend", min_value=0.0, value=500000.0, format="%.2f")
period = st.sidebar.selectbox("Select Profit Period", ["Quarterly", "Semi-Annual", "Annual"])

# Create a session state to store prediction data
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_value = 0
    st.session_state.rd_value = 0
    st.session_state.admin_value = 0
    st.session_state.marketing_value = 0
    st.session_state.period = ""

if st.sidebar.button("Calculate profit"):
    input_df = pd.DataFrame([[rd, admin, marketing]], columns=["R&D Spend", "Administration", "Marketing Spend"])
    prediction = model.predict(input_df)[0]
    
    # Adjust profit based on period
    if period == "Quarterly":
        adjusted_profit = prediction / 4
    elif period == "Semi-Annual":
        adjusted_profit = prediction / 2
    else:
        adjusted_profit = prediction  # Annual by default
    
    st.sidebar.success(f"{period} Profit: {format_currency(adjusted_profit)}")
    
    # Store in session state
    st.session_state.prediction_made = True
    st.session_state.prediction_value = adjusted_profit
    st.session_state.rd_value = rd
    st.session_state.admin_value = admin
    st.session_state.marketing_value = marketing
    st.session_state.period = period

st.markdown("<h1 style='font-size: 32px;'>Predicted Model Using Linear Regression</h1>", unsafe_allow_html=True)
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    required_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit", "State"]
    if not all(col in data.columns for col in required_cols):
      st.error("File missing required columns.")
    else:
        try:
            X = data[["R&D Spend", "Administration", "Marketing Spend"]]
            y_test = data["Profit"]
            y_pred = model.predict(X)
            
            df = pd.DataFrame({
                'Profit': y_test,
                'Predicted Profit': y_pred.flatten(),
                'State': data['State'],
                'R&D Spend': data['R&D Spend'],
                'Administration': data['Administration'],
                'Marketing Spend': data['Marketing Spend']
            })    
            
            #metrics
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                a1.metric(
                    "Maximum Profit", 
                    format_currency(df['Profit'].max()),
                    help="Highest profit value in the dataset"
                )

            with a2:
                a2.metric(
                    "Minimum Profit", 
                    format_currency(df['Profit'].min()),
                    help="Lowest profit value in the dataset"
                )

            with a3:
                a3.metric(
                    "Maximum Predicted Profit", 
                    format_currency(df['Predicted Profit'].max()),
                    help="Highest predicted profit value"
                )

            with a4:
                min_value = df[df['Predicted Profit'] > 0]['Predicted Profit'].min() if not df[df['Predicted Profit'] > 0].empty else 0
                a4.metric(
                    "Minimum Predicted Profit", 
                    format_currency(min_value),
                    help="Lowest positive predicted profit value"
                )

            # Additional metrics for averages
            b1, b2, b3, b4 = st.columns(4)
            
            with b1:
                b1.metric(
                    "Average R&D Spend", 
                    format_currency(df['R&D Spend'].mean()),
                    help="Average R&D expenditure across all companies"
                )
            
            with b2:
                b2.metric(
                    "Average Administration", 
                    format_currency(df['Administration'].mean()),
                    help="Average administration costs across all companies"
                )
            
            with b3:
                b3.metric(
                    "Average Marketing Spend", 
                    format_currency(df['Marketing Spend'].mean()),
                    help="Average marketing expenditure across all companies"
                )
            
            with b4:
                b4.metric(
                    "Average Profit", 
                    format_currency(df['Profit'].mean()),
                    help="Average profit across all companies"
                )

            # shown actual data
            with st.expander("Shown Data"):
                st.dataframe(data.style.format({
                    'R&D Spend': '{:,.0f}',
                    'Administration': '{:,.0f}',
                    'Marketing Spend': '{:,.0f}',
                    'Profit': '{:,.0f}'
                }))


            # actual vs pred df
            st.write("Actual vs Predictible Profit")
            df2 = pd.DataFrame({
                'Profit': y_test,
                'Predicted Profit': y_pred.flatten(),
            })
            with st.expander("Shown Predicted vs Actual Data"):
                st.dataframe(df2.style.format({
                    'Profit': '{:,.0f}',
                    'Predicted Profit': '{:,.0f}'
                }))

            pred = pd.DataFrame({'Predicted profit': y_pred})
            data_with_pred = pd.concat([data, pred], axis=1)

            # graphics
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Graphs", "States comparison", "States Distribution", "Profit Forecast", "Model Performance"])

            with tab1:
                st.subheader("Actual vs Predicted Profit by State")
                fig = px.scatter(
                    df,
                    x='Profit',
                    y='Predicted Profit',
                    color=fil2 if fil2 else 'State',
                    size=fil1,
                    facet_col='State',
                    labels={'Profit': 'Actual Profit', 'Predicted Profit': 'Predicted Profit'},
                    template=chart_theme,
                    color_discrete_sequence=color_palette,
                    title="Actual vs Predicted Profit by State"
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    title_font=dict(size=20),
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    height=500
                )
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                
                # Format axis labels with commas for thousands
                fig.update_xaxes(tickformat=",")
                fig.update_yaxes(tickformat=",")
                
                st.plotly_chart(fig, use_container_width=True)
                c1, c2, c3 = st.columns((4, 3, 3))

                with c1:
                    fig1 = px.bar(
                        df, 
                        x='State', 
                        y='Profit', 
                        color=fil2 if fil2 else 'State',
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs Profit",
                        text_auto=True
                    )
                    fig1.update_layout(
                        xaxis_tickangle=-45,
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        yaxis_title="Profit (SAR)",
                        height=400,
                        xaxis_title=""
                    )
                    fig1.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                    fig1.update_yaxes(tickformat=",")
                    st.plotly_chart(fig1, use_container_width=True)

                with c2:
                    fig2 = px.bar(
                        df, 
                        x='State', 
                        y='Marketing Spend', 
                        color=fil2 if fil2 else 'State',
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs Marketing Spend",
                        text_auto=True
                    )
                    fig2.update_layout(
                        xaxis_tickangle=-45,
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        yaxis_title="Marketing Spend (SAR)",
                        height=400,
                        xaxis_title=""
                    )
                    fig2.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                    fig2.update_yaxes(tickformat=",")
                    st.plotly_chart(fig2, use_container_width=True)

                with c3:
                    fig3 = px.bar(
                        df, 
                        x='State', 
                        y='Administration', 
                        color=fil2 if fil2 else 'State',
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs Administration",
                        text_auto=True
                    )
                    fig3.update_layout(
                        xaxis_tickangle=-45,
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        yaxis_title="Administration (SAR)",
                        height=400,
                        xaxis_title=""
                    )
                    fig3.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                    fig3.update_yaxes(tickformat=",")
                    st.plotly_chart(fig3, use_container_width=True)

            #row c
            with tab2:
                st.subheader("Predicted Profit by State")
                d1, d2, d3 = st.columns((4, 3, 3))

                with d1:
                    fig = px.pie(
                        df, 
                        names="State", 
                        values="Predicted Profit", 
                        hole=0.4,
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs Predicted Profit"
                    )
                    fig.update_layout(
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    fig.update_traces(textinfo='percent+label', textposition='inside')
                    st.plotly_chart(fig, use_container_width=True)

                with d2:
                    fig = px.pie(
                        df, 
                        names="State", 
                        values="Marketing Spend", 
                        hole=0.4,
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs Marketing Spend"
                    )
                    fig.update_layout(
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    fig.update_traces(textinfo='percent+label', textposition='inside')
                    st.plotly_chart(fig, use_container_width=True)

                with d3:
                    fig = px.pie(
                        df, 
                        names="State", 
                        values="R&D Spend", 
                        hole=0.4,
                        template=chart_theme,
                        color_discrete_sequence=color_palette,
                        title="State vs R&D Spend"
                    )
                    fig.update_layout(
                        title_font=dict(size=18),
                        paper_bgcolor=chart_bg_color,
                        plot_bgcolor=chart_bg_color,
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    fig.update_traces(textinfo='percent+label', textposition='inside')
                    st.plotly_chart(fig, use_container_width=True)
            with tab3:
                st.subheader("Data Summary and Statistics")
                
                # Format numbers in the summary statistics
                st.markdown("### Summary Statistics")
                st.dataframe(df.describe().style.format({
                    'Profit': '{:,.0f}',
                    'Predicted Profit': '{:,.0f}',
                    'R&D Spend': '{:,.0f}',
                    'Administration': '{:,.0f}',
                    'Marketing Spend': '{:,.0f}'
                }))
                
                # Show processed data with proper formatting
                st.markdown("### Complete Dataset")
                st.dataframe(df.style.format({
                    'Profit': '{:,.0f}',
                    'Predicted Profit': '{:,.0f}',
                    'R&D Spend': '{:,.0f}',
                    'Administration': '{:,.0f}',
                    'Marketing Spend': '{:,.0f}'
                }))
                
                # Add correlation heatmap
                st.markdown("### Correlation Heatmap")
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                corr = numeric_df.corr()
                
                fig = px.imshow(
                    corr, 
                    text_auto=True, 
                    color_continuous_scale=px.colors.sequential.Blues,
                    template=chart_theme,
                    title="Correlation Between Variables"
                )
                fig.update_layout(
                    height=500,
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    title_font=dict(size=18)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for the data
                st.markdown("### Download Data")
                
                def convert_df_to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Predictions')
                    processed_data = output.getvalue()
                    return processed_data

                excel_file = convert_df_to_excel(df)
                
                st.download_button(
                    label="Download Excel File",
                    data=excel_file,
                    file_name='predicted_profit.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            # User Profit Forecast tab
            with tab4:
                if st.session_state.prediction_made:
                    st.markdown(f"### {st.session_state.period} Profit Prediction Results")
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted Profit", 
                            format_currency(st.session_state.prediction_value),
                            help=f"Predicted {st.session_state.period.lower()} profit based on your inputs"
                        )
                    
                    # Create a comparison dataframe for visualization
                    comparison_df = pd.DataFrame({
                        'Category': ['R&D Spend', 'Administration', 'Marketing Spend'],
                        'Amount': [st.session_state.rd_value, st.session_state.admin_value, st.session_state.marketing_value]
                    })
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Create a bar chart showing the input values
                        fig = px.bar(
                            comparison_df, 
                            x='Category', 
                            y='Amount',
                            color='Category',
                            text_auto=True,
                            template=chart_theme,
                            color_discrete_sequence=color_palette,
                            title="Budget Allocation"
                        )
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="Amount (SAR)",
                            paper_bgcolor=chart_bg_color,
                            plot_bgcolor=chart_bg_color,
                            height=400,
                            title_font=dict(size=18),
                            showlegend=False
                        )
                        fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                        fig.update_yaxes(tickformat=",")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col4:
                        # Create a pie chart showing the budget distribution
                        fig2 = px.pie(
                            comparison_df,
                            values='Amount',
                            names='Category',
                            hole=0.4,
                            template=chart_theme,
                            color_discrete_sequence=color_palette,
                            title="Budget Distribution"
                        )
                        fig2.update_layout(
                            paper_bgcolor=chart_bg_color,
                            plot_bgcolor=chart_bg_color,
                            height=400,
                            title_font=dict(size=18),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        fig2.update_traces(textinfo='percent+label', textposition='inside')
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown(f"""
                    ### Analysis
                    Based on your input:
                    - R&D Spend: {format_currency(st.session_state.rd_value)}
                    - Administration: {format_currency(st.session_state.admin_value)}
                    - Marketing Spend: {format_currency(st.session_state.marketing_value)}
                    
                    The predicted {st.session_state.period.lower()} profit is **{format_currency(st.session_state.prediction_value)}**
                    """)
                    
                    # Add a section to compare with existing data
                    if 'df' in locals():
                        st.markdown("### How Your Forecast Compares")
                        avg_profit = df['Profit'].mean()
                        comparison = (st.session_state.prediction_value / avg_profit) * 100 if st.session_state.period == "Annual" else (st.session_state.prediction_value * (4 if st.session_state.period == "Quarterly" else 2) / avg_profit) * 100
                        
                        comparison_text = "above average" if comparison > 100 else "below average"
                        st.info(f"Your predicted annual profit is {abs(comparison-100):.1f}% {comparison_text} compared to the dataset average of {format_currency(avg_profit)}.")
                else:
                    st.info("Use the sidebar to calculate profit predictions and see visualizations here.")

            # Model Performance tab
            with tab5:
                st.subheader("Model Performance Metrics")
                
                # Calculate performance metrics
                y_true = df['Profit']
                y_pred = df['Predicted Profit']
                
                # R-squared (coefficient of determination)
                r2 = r2_score(y_true, y_pred)
                
                # Mean Squared Error
                mse = mean_squared_error(y_true, y_pred)
                
                # Root Mean Squared Error
                rmse = np.sqrt(mse)
                
                # Mean Absolute Error
                mae = mean_absolute_error(y_true, y_pred)
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # Adjusted R-squared
                n = len(y_true)
                p = 3  # number of predictors (R&D, Admin, Marketing)
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                
                # Display metrics in columns
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("R² Score", f"{r2:.4f}", 
                              help="Coefficient of determination: Higher is better (1.0 is perfect fit)")
                    st.metric("Adjusted R²", f"{adjusted_r2:.4f}", 
                              help="R² adjusted for number of predictors")
                    st.metric("Mean Absolute Error (MAE)", f"{format_currency(mae)}", 
                              help="Average absolute difference between predicted and actual values")
                
                with metrics_col2:
                    st.metric("Mean Squared Error (MSE)", f"{format_currency(mse)}", 
                              help="Average squared difference between predicted and actual values")
                    st.metric("Root Mean Squared Error (RMSE)", f"{format_currency(rmse)}", 
                              help="Square root of MSE, in the same units as the target variable")
                    st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%", 
                              help="Average percentage difference between predicted and actual values")
                
                # Visualization of actual vs predicted values
                st.subheader("Actual vs Predicted Values")
                
                # Create a dataframe for regression line
                min_val = min(y_true.min(), y_pred.min()) * 0.9
                max_val = max(y_true.max(), y_pred.max()) * 1.1
                line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})
                
                # Create scatter plot with regression line
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=y_true, 
                    y=y_pred, 
                    mode='markers',
                    marker=dict(
                        color='rgba(0, 128, 255, 0.8)',
                        size=10,
                        line=dict(
                            color='rgba(0, 0, 0, 0.5)',
                            width=1
                        )
                    ),
                    name='Predictions'
                ))
                
                # Add perfect prediction line
                fig.add_trace(go.Scatter(
                    x=line_df['x'],
                    y=line_df['y'],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.8)', dash='dash'),
                    name='Perfect Prediction'
                ))
                
                # Update layout
                fig.update_layout(
                    title='Actual vs Predicted Profit',
                    xaxis_title='Actual Profit (SAR)',
                    yaxis_title='Predicted Profit (SAR)',
                    template=chart_theme,
                    height=600,
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Format axes with commas
                fig.update_xaxes(tickformat=",")
                fig.update_yaxes(tickformat=",")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual Plot
                st.subheader("Residual Analysis")
                
                # Calculate residuals
                residuals = y_true - y_pred
                
                # Create residual plot
                fig_residual = go.Figure()
                
                # Add residual scatter plot
                fig_residual.add_trace(go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    marker=dict(
                        color='rgba(0, 128, 255, 0.8)',
                        size=10,
                        line=dict(
                            color='rgba(0, 0, 0, 0.5)',
                            width=1
                        )
                    ),
                    name='Residuals'
                ))
                
                # Add zero line
                fig_residual.add_trace(go.Scatter(
                    x=[min(y_pred) * 0.9, max(y_pred) * 1.1],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.8)', dash='dash'),
                    name='Zero Line'
                ))
                
                # Update layout
                fig_residual.update_layout(
                    title='Residual Plot (Predicted vs Error)',
                    xaxis_title='Predicted Profit (SAR)',
                    yaxis_title='Residual (Actual - Predicted)',
                    template=chart_theme,
                    height=600,
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Format x-axis with commas
                fig_residual.update_xaxes(tickformat=",")
                fig_residual.update_yaxes(tickformat=",")
                
                st.plotly_chart(fig_residual, use_container_width=True)
                
                # Residual Distribution
                st.subheader("Residual Distribution")
                
                # Create histogram of residuals
                fig_hist = px.histogram(
                    residuals, 
                    nbins=20, 
                    title="Distribution of Residuals",
                    labels={'value': 'Residual Value', 'count': 'Frequency'},
                    template=chart_theme,
                    color_discrete_sequence=['rgba(0, 128, 255, 0.8)']
                )
                
                # Update layout
                fig_hist.update_layout(
                    height=400,
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    title_font=dict(size=18),
                    showlegend=False
                )
                
                # Format x-axis with commas
                fig_hist.update_xaxes(tickformat=",")
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                
                # Get coefficients from the model
                coef = model.coef_
                
                # Create dataframe for feature importance
                features = ['R&D Spend', 'Administration', 'Marketing Spend']
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': coef,
                    'Absolute Coefficient': np.abs(coef)
                })
                
                # Sort by absolute coefficient
                importance_df = importance_df.sort_values('Absolute Coefficient', ascending=False)
                
                # Create bar chart
                fig_importance = px.bar(
                    importance_df,
                    x='Feature',
                    y='Coefficient',
                    title='Feature Importance (Coefficients)',
                    color='Coefficient',
                    color_continuous_scale=px.colors.sequential.Blues,
                    template=chart_theme
                )
                
                # Update layout
                fig_importance.update_layout(
                    height=400,
                    paper_bgcolor=chart_bg_color,
                    plot_bgcolor=chart_bg_color,
                    title_font=dict(size=18),
                    xaxis_title="",
                    yaxis_title="Coefficient Value"
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Model explanation
                st.subheader("Model Interpretation")
                
                # Get intercept and coefficients
                intercept = model.intercept_
                coefficients = {feature: coef for feature, coef in zip(features, model.coef_)}
                
                # Create mathematical formula
                formula = f"Profit = {intercept:.2f}"
                for feature, coef in coefficients.items():
                    if coef >= 0:
                        formula += f" + {coef:.4f} × {feature}"
                    else:
                        formula += f" - {abs(coef):.4f} × {feature}"
                
                # Display the formula
                st.markdown(f"""
                ### Linear Regression Formula
                
                ```
                {formula}
                ```
                
                ### Interpretation
                
                - **R&D Spend**: For each additional SAR 1 spent on R&D, the profit increases by approximately SAR {coefficients['R&D Spend']:.4f}
                - **Administration**: For each additional SAR 1 spent on Administration, the profit changes by approximately SAR {coefficients['Administration']:.4f}
                - **Marketing Spend**: For each additional SAR 1 spent on Marketing, the profit increases by approximately SAR {coefficients['Marketing Spend']:.4f}
                - **Baseline**: Without any spending, the expected profit would be SAR {intercept:.2f}
                
                ### Model Quality Assessment
                
                - This model explains approximately **{r2*100:.2f}%** of the variance in profit.
                - The model has a mean absolute percentage error (MAPE) of **{mape:.2f}%**.
                """)

        except Exception as e:
            st.error(f"Error occurred: {e}")

if not uploaded_file:
    st.write("Please Upload file") 