#### Evaluation functions for tree model ####
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.tree import export_graphviz
import graphviz

def evaluate_model(model, X_train, y_train, X_test, y_test):

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Print metrics
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("Specificity:", tn / (tn + fp))
    print("F1 Score:", f1_score(y_test, y_test_pred))
    print("F2 Score:", fbeta_score(y_test, y_test_pred, beta=2))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        display_labels=["Not Cancelled", "Cancelled"]
    )

    plt.title("Confusion Matrix")
    plt.show()


def visualize_tree(tree_model, X_train, model_name="tree_model"):
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=X_train.columns,
        class_names=["Not Cancelled", "Cancelled"],
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3
    )

    graph = graphviz.Source(dot_data)

    # graph.render(model_name, format="png", cleanup=True) # uncomment this for storing the tree as a PNG file
    
    return graph

def visualize_feature_importance(tree_model, X):
    importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.head(5)
    top5 = importance_df.head(5)
    print(top5)

    plt.figure(figsize=(8, 5))
    plt.barh(top5["Feature"][::-1], top5["Importance"][::-1])

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 5 Feature Importances")

    plt.tight_layout()
    plt.show()

def plot_roc_comparison(model1, model2, X_test, y_test, model1_name="Model 1", model2_name="Model 2"):
    # Predicted probabilities
    prob1 = model1.predict_proba(X_test)[:, 1]
    prob2 = model2.predict_proba(X_test)[:, 1]

    # ROC values
    fpr1, tpr1, _ = roc_curve(y_test, prob1)
    fpr2, tpr2, _ = roc_curve(y_test, prob2)

    # AUC scores
    auc1 = roc_auc_score(y_test, prob1)
    auc2 = roc_auc_score(y_test, prob2)

    # Plot
    plt.figure(figsize=(8, 6))

    plt.plot(fpr1, tpr1, label=f"{model1_name} (AUC = {auc1:.3f})")

    plt.plot(fpr2, tpr2, label=f"{model2_name} (AUC = {auc2:.3f})")

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()

#### Evaluation functions for regression model ####

#### Evaluation functions for CNN ####