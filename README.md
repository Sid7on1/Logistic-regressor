# ****🔍 Logistic Regression Diabetes Predictor****

This project uses **Logistic Regression**, a supervised classification algorithm, to predict whether a patient has diabetes based on medical attributes.

## 🧪 Is it a Diabetes Tester?

✅ Yes! This model acts like a diabetes test predictor. It takes in features like glucose level, insulin, BMI, age, and more, and outputs:

- `0` = No diabetes  
- `1` = Has diabetes  

The model is trained on the **Pima Indians Diabetes Dataset**, which is a real-world dataset often used for medical ML tasks.

## ⚙️ What Happens in the Code?

1. 📥 Loads dataset from a public CSV  
2. 🧹 Splits it into features (`X`) and labels (`y`)  
3. 🧪 Trains the Logistic Regression model  
4. 🔮 Predicts whether the test data has diabetes  
5. 📈 Evaluates model using accuracy, confusion matrix, and classification report  
6. 🎨 Visualizes confusion matrix with Seaborn heatmap  

> Built using Python, Pandas, scikit-learn, and Matplotlib.
