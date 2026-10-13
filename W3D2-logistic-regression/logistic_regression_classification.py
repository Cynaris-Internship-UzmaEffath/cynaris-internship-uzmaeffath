# W3D2: Logistic Regression and Classification
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,RocCurveDisplay)
from pathlib import Path
output =Path(__file__).parent / "outputs"
data=load_breast_cancer()
X=data.data
y=data.target
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]
print("Logistic Regression - Breast Cancer\n")
print("\nIntercept\n:",model.named_steps["logisticregression"].intercept_)
print("\nCoefficients:",model.named_steps["logisticregression"].coef_)
print("\n accuarcy:",accuracy_score(y_test,y_pred))
print("\nprecision:", precision_score(y_test,y_pred))
print("\nrecall:",recall_score(y_test,y_pred))
print("\n F1-Score:",f1_score(y_test,y_pred))
print("\nROC-AUC:",roc_auc_score(y_test,y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test,y_pred,target_names=data.target_names))
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=data.target_names,yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(output/ "confusion_matrix.png")
plt.close()
RocCurveDisplay.from_predictions(y_test,y_prob)
plt.title(f"ROC Curve(AUC={roc_auc_score(y_test,y_prob):.3f})")
plt.tight_layout()
plt.savefig(output/ "roc_curve.png")
plt.close()
print("\n IRIS MULTICLASS CLASSIFICATION!!!!!!!!!")

# 6. Multiclass classification using Iris dataset

print("\nIris Multiclass Classification")

iris = load_iris()

X_iris = iris.data
y_iris = iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris
)

models = {
    "One-vs-Rest": make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=5000))
    ),
    "Softmax": make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=5000)
    )
}

results = []

for name, classifier in models.items():
    classifier.fit(X_train_iris, y_train_iris)

    iris_pred = classifier.predict(X_test_iris)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_iris, iris_pred),
        "Precision": precision_score(
            y_test_iris, iris_pred, average="macro"
        ),
        "Recall": recall_score(
            y_test_iris, iris_pred, average="macro"
        ),
        "F1 Score": f1_score(
            y_test_iris, iris_pred, average="macro"
        )
    })

comparison = pd.DataFrame(results)

print("\nOne-vs-Rest vs Softmax:")
print(comparison.round(3).to_string(index=False))

comparison.to_csv(output / "iris_comparison.csv", index=False)

print("\nAll tasks completed successfully!")
