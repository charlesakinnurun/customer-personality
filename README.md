# Customer Personality
![Customers](/customer.jpg)
In today’s data-driven environment, understanding who your customers are is just as important as understanding what they buy. The customer-personality project is designed to bridge that gap: to explore, model and predict distinct personality profiles of customers based on their behavioural and transaction data. By doing so, the project aims to empower businesses, marketers and data scientists with deeper insights into customer segments — enabling more personalised engagement, tailored messaging and stronger customer relationships.

## Procedures
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
- New Prediction Input Function
<<<<<<< HEAD

=======
>>>>>>> 655e27a1c4b81a133b66b38351627fd8bbbf77f1

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
## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE)
 file for details.
