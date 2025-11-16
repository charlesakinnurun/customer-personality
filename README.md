# Customer Personality
![Customers](/customer.jpg)
In today’s data-driven environment, understanding who your customers are is just as important as understanding what they buy. The customer-personality project is designed to bridge that gap: to explore, model and predict distinct personality profiles of customers based on their behavioural and transaction data. By doing so, the project aims to empower businesses, marketers and data scientists with deeper insights into customer segments — enabling more personalised engagement, tailored messaging and stronger customer relationships.

## Procedures
- Import Libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Loading
- Data Preprocessing
    - Handle missing values
    - Handle duplicated rows
    - Drop missing values
- Feature Selection and Engineering
- Data Scaling
    - StandardScaler
- Dimensionality Reduction
    - Principal Component Analysis
- Pre-Training Visualization
    - Data Distribution using scatterplot
![pre-training-visualization](/output1.png)
- Hyperparameter Tuning
    - Optimal K determination using Silhouette Score
- Model Comparison
    - K-Means
    - Aggolomerative
    - DBSCAN
    - Gaussian Mixture Model (GMM)
- Model Training
- Model Evaluation
    - Silhouette Score
    - Calinski Harabaz Sore
- Post-Training Visualization
![post-training-visualiztion](/output2.png)
- New Prediction Input Function


## Process
![Screenshot(228)](/Screenshot%20(228).png)
![Screenshot(229)](/Screenshot%20(229).png)
![Screenshot(230)](/Screenshot%20(230).png)
![Screenshot(231)](/Screenshot%20(231).png)
![Screenshot(232)](/Screenshot%20(232).png)
![Screenshot(233)](/Screenshot%20(233).png)


## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/customer-personality.git
cd customer-personality
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
customer-personality/
│
├── model.ipynb  
|── model.py    
|── marketing_campaign.csv  
├── requirements.txt 
├── customer.jpg       
├── output1.png        
├── output2.png        
├── Screenshot (228).png
├── Screenshot (229).png
├── Screenshot (230).png
├── Screenshot (231).png
├── Screenshot (232).png
├── Screenshot (233).png
├── SECURITY.md        
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```
## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```
