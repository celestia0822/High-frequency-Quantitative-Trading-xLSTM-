# RF-PCA-xLSTM: High-Frequency Futures Price Forecasting

## Overview
High-frequency futures price forecasting plays a critical role in quantitative trading, risk management, and market stability. However, high-frequency financial time series are typically noisy, non-stationary, and highly nonlinear, posing significant challenges for traditional statistical models.

This project proposes an integrated forecasting framework—**RF-PCA-xLSTM**—that combines **Random Forest (RF)** for feature selection, **Principal Component Analysis (PCA)** for dimensionality reduction, and the **extended Long Short-Term Memory (xLSTM)** network for sequence modeling. The framework is designed to improve prediction accuracy and stability when modeling multi-dimensional, high-frequency futures data.

## Key Contributions
- Constructed a **high-frequency futures dataset** with second-level data aggregated to minute frequency, covering **11 actively traded futures contracts** across financial and commodity markets.
- Identified **cross-market interactions** among futures prices using **correlation analysis and Granger causality tests**, and incorporated cross-market features into the indicator pool.
- Proposed the **RF-PCA-xLSTM** framework, integrating feature selection, dimensionality reduction, and advanced sequence modeling.
- Demonstrated significant improvements over traditional deep learning models (GRU, LSTM), achieving:
  - ~40% reduction in RMSE  
  - ~25% reduction in MSPE  
  - ~13% reduction in MAPE  
  - ~1.5% increase in R²

## Data Description
- **Source:** High-frequency futures market data provided by competition organizers  
- **Time Period:** April 1–30, 2024  
- **Frequency:** Original data at sub-second resolution, aggregated to **1-minute intervals**  
- **Contracts:** 11 futures including stock index futures and commodity futures  
- **Features:**
  - Basic trading indicators (open, high, low, volume, open interest, etc.)
  - Technical indicators (VWAP, spreads, SMA, Bollinger Bands, order book imbalance)
  - Cross-market indicators derived from causality analysis

## Methodology

### 1. Feature Engineering
- Constructed a multi-category indicator pool:
  - Basic trading indicators  
  - Technical indicators  
  - Cross-market influence indicators
- Applied **Random Forest** to rank feature importance and select key predictors.

### 2. Dimensionality Reduction
- Used **PCA** to reduce multicollinearity and retain principal components explaining ≥90% of variance.

### 3. Time-Series Modeling
- Implemented **xLSTM**, an extension of traditional LSTM that integrates:
  - sLSTM (scalar memory with exponential gating)
  - mLSTM (matrix memory with covariance-based updates)
- Compared performance across:
  - RNN
  - MLP
  - GRU
  - LSTM
  - xLSTM
  - RF-PCA-xLSTM

### 4. Evaluation Strategy
- Rolling-window forecasting to simulate real trading conditions  
- Dataset split: 80% training, 10% validation, 10% testing  
- Evaluation metrics:
  - RMSE, MAE, MSPE, MAPE  
  - R² (coefficient of determination)  
  - Directional prediction statistic (Dstat)  
  - Iteration time

## Results
- **Single-variable xLSTM** significantly outperformed GRU and LSTM models, demonstrating strong stability and long-term dependency modeling.
- **Multi-variable RF-PCA-xLSTM** further improved accuracy and robustness by leveraging cross-market information and reduced feature redundancy.
- The proposed framework achieved smoother predictions and better trend-following behavior under high volatility conditions.

## Tools & Environment
- **Language:** Python 3.9  
- **Framework:** PyTorch  
- **Libraries:** NumPy, pandas, scikit-learn, pytorch  
- **Hardware:** NVIDIA RTX 4060 GPU  
- **OS:** Windows 10  

## Applications
- High-frequency futures price forecasting  
- Quantitative trading strategy support  
- Market risk analysis  
- Extension to:
  - Equity price prediction  
  - Government bond volatility forecasting  

## Future Work
- Compare RF-PCA-xLSTM with **Transformer** and **state-space models (e.g., Mamba)**.
- Extend the framework to additional financial markets and longer time horizons.
- Incorporate more advanced feature construction and adaptive window strategies.

## References
This project builds upon recent advances in deep learning for time-series forecasting, including xLSTM (Beck et al., 2024), and established literature in high-frequency financial modeling.
