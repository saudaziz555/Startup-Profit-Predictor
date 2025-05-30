# Startup Profit Predictor

A machine learning web application that predicts startup profits based on spending in R&D, Administration, and Marketing.

![Startup Profit Predictor](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## Overview

This application uses a Linear Regression model to predict a startup's profit based on its spending in three key areas: R&D, Administration, and Marketing. The model was trained on a dataset of 50 startups and their corresponding profits.

## Features

- **Profit Prediction**: Enter your budget allocation to get instant profit forecasts
- **Data Visualization**: Interactive charts and graphs showing the relationship between spending and profits
- **Model Performance**: Comprehensive metrics and visualizations of the model's accuracy
- **Data Analysis**: State-wise comparison and distribution of profits and expenses
- **Download Results**: Export your prediction results to Excel

## Screenshots

![App Screenshot](https://via.placeholder.com/800x400?text=App+Screenshot)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/startup-profit-predictor.git
cd startup-profit-predictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Data (Optional)**: Upload your own Excel file with startup data for analysis
2. **Input Values**: Enter your R&D, Administration, and Marketing expenses
3. **Calculate Profit**: Get profit predictions and visualizations
4. **Explore Analytics**: Navigate through the different tabs to explore the data and predictions
5. **Download Results**: Export the predictions to Excel for further analysis

## Data Format

To use your own data, make sure your Excel file has the following columns:
- R&D Spend
- Administration
- Marketing Spend
- State
- Profit

## Model Details

- **Algorithm**: Linear Regression
- **Features**: R&D Spend, Administration, Marketing Spend
- **Target**: Profit
- **Performance Metrics**: R², MSE, RMSE, MAE, and MAPE

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset source: [50 Startups Data](https://www.kaggle.com/datasets/farhanmd29/50-startups)
- Built with [Streamlit](https://streamlit.io/)
- Created by [Saud Alswaeh]

## Contact

For any inquiries, please open an issue on this repository or contact [salswaeh@gmail.com]. 