import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# CONFIGURACIÓN STREAMLIT
# --------------------------

st.set_page_config(page_title="Iris ML Classifier", layout="wide")
st.title("🌸 Iris Dataset - Clasificación con Múltiples Modelos")

# --------------------------
# CARGAR DATA
# --------------------------

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Sidebar
st.sidebar.header("Configuración del Modelo")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

model_option = st.sidebar.selectbox(
    "Selecciona el Modelo",
    (
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest"
    )
)

metric_option = st.sidebar.multiselect(
    "Selecciona Métricas",
    ("Accuracy", "Precision", "Recall", "F1-Score"),
    default=["Accuracy"]
)

visual_option = st.sidebar.multiselect(
    "Opciones de Visualización",
    (
        "Matriz de Confusión",
        "Curva ROC",
        "Frontera de Decisión"
    ),
    default=["Matriz de Confusión"]
)

# --------------------------
# PREPROCESAMIENTO
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# MODELOS
# --------------------------

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)

elif model_option == "SVM":
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf"))
    model = SVC(C=C, kernel=kernel, probability=True)

elif model_option == "KNN":
    k = st.sidebar.slider("Número de Vecinos (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth)

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de Árboles", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators)

# --------------------------
# ENTRENAMIENTO
# --------------------------

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------
# MÉTRICAS
# --------------------------

st.subheader("📊 Métricas de Desempeño")

metrics_dict = {}

if "Accuracy" in metric_option:
    metrics_dict["Accuracy"] = accuracy_score(y_test, y_pred)

if "Precision" in metric_option:
    metrics_dict["Precision"] = precision_score(y_test, y_pred, average="weighted")

if "Recall" in metric_option:
    metrics_dict["Recall"] = recall_score(y_test, y_pred, average="weighted")

if "F1-Score" in metric_option:
    metrics_dict["F1-Score"] = f1_score(y_test, y_pred, average="weighted")

st.write(pd.DataFrame(metrics_dict, index=["Valor"]))

# --------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------

if "Matriz de Confusión" in visual_option:
    st.subheader("🔎 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig)

# --------------------------
# CURVA ROC (OvR)
# --------------------------

if "Curva ROC" in visual_option:
    st.subheader("📈 Curva ROC (One vs Rest)")

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots()

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(fig)

# --------------------------
# FRONTERA DE DECISIÓN
# --------------------------

if "Frontera de Decisión" in visual_option:
    st.subheader("🌐 Frontera de Decisión (PCA 2D)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y, test_size=test_size, random_state=42, stratify=y
    )

    model.fit(X_train_pca, y_train_pca)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor="k")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot(fig)

# --------------------------
# FOOTER
# --------------------------

st.markdown("---")
st.markdown("Desarrollado con Streamlit 🚀")