# Forecasting Stock Prices Using LSTM Networks

This project explores the use of deep learning, particularly **Long Short-Term Memory (LSTM)** networks, for forecasting stock prices. It compares LSTMs with traditional linear models and incorporates financial fundamentals from CRSP and Compustat databases. The work was conducted using real S&P 500 data accessed via WRDS.

## Approach Overview

This project implements a **stock price prediction pipeline** using deep learning, focusing on the **S&P 500 index**. The approach compares traditional linear models with more advanced **Long Short-Term Memory (LSTM)** networks.

### 1. Data Acquisition & Preprocessing
- **Source**: [Wharton Research Data Services (WRDS)](https://wrds-www.wharton.upenn.edu/)  
- **Databases used**:  
  - **CRSP**: stock prices, returns, volume  
  - **Compustat**: financial indicators like book value, liabilities, equity  
- **Merging**: CRSP and Compustat datasets are aligned via the **CCM Link Table**  
- **Cleaning & Feature Engineering**: includes adjusted prices, book-to-market ratios, and other financial metrics

### 2. Exploratory Data Analysis
- Distribution of prices and volumes  
- Visualization of price trends by company, industry, and exchange  
- Outlier handling and market structure insights  

### 3. Baseline Models
- **Linear Regression**
- **Lasso Regression**  
Tested with:
  - Random vs time-based train/test splits  
  - Segmenting high vs low priced stocks  
  - Feature selection and error metrics (MSE, MAE, R², etc.)

### 4. LSTM Architectures
- Implemented using **PyTorch**
- Trained to predict future prices based on historical sequences
- Two model variants:
  - **Univariate LSTM** (just past prices)
  - **Multivariate LSTM** (prices + volume + financial ratios)

### 5. LSTM Enhancements
- **Gradient Clipping**  
- **Early Stopping**  
- **Hyperparameter tuning** (hidden size, model depth, sequence length)  
- Visualized training/test RMSE curves & plotted predictions

### 6. Stress Testing
- Evaluated model on:
  - Stocks with **extreme prices** (e.g. NVR Inc.)
  - Stocks with **extreme volumes** (e.g. Bank of America)
  - Stocks with **limited history** (e.g. Sprint Nextel)


## Repository Structure

```bash
├── Notebook.ipynb              # Main Jupyter notebook with LSTM modeling
├── Report.pdf                  # Thesis report
├── plots/                      # Visualizations for stock JNJ (multivariate LSTM)
│   ├── predictions.png         # Predicted vs actual stock prices
│   └── error.png               # RMSE over training epochs
└── README.md
```


## Key Insights

- LSTM models outperformed linear models in capturing financial time series patterns  
- Gradient clipping and early stopping improved stability and convergence  
- Multivariate LSTMs offer richer representations but may introduce noise  
- Model performance varied significantly depending on stock characteristics  


## References

All data was obtained via [WRDS](https://wrds.wharton.upenn.edu/) using academic access to **CRSP** and **Compustat** databases. The thesis was supervised by **Prof. Chen Liu** and conducted in collaboration with City University of Hong Kong during a research exchange.
