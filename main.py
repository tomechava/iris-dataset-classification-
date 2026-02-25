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

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Iris ML Classifier", layout="wide")
st.title("🌸 Iris Dataset - Clasificación Interactiva")

# --------------------------------------------------
# DATA
# --------------------------------------------------

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
feature_names = iris.feature_names

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Configuración del Modelo")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

model_option = st.sidebar.selectbox(
    "Modelo",
    (
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest"
    )
)

# --------------------------------------------------
# SPLIT + SCALE
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODELOS
# --------------------------------------------------

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)

elif model_option == "SVM":
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf"))
    model = SVC(C=C, kernel=kernel, probability=True)

elif model_option == "KNN":
    k = st.sidebar.slider("K", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth)

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de Árboles", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators)

# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------------------------------
# MÉTRICAS GENERALES
# --------------------------------------------------

st.subheader("📊 Desempeño General del Modelo")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
col4.metric("F1", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

# --------------------------------------------------
# PREDICCIÓN MANUAL
# --------------------------------------------------

st.markdown("---")
st.subheader("🌼 Predicción Manual")

st.write("Ingrese los valores de la flor:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

real_class_option = st.selectbox(
    "Clase real (opcional para calcular error)",
    ("No especificar",) + tuple(class_names)
)

if st.button("Predecir"):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)[0]
    probabilities = model.predict_proba(user_input_scaled)[0]

    st.success(f"Predicción: **{class_names[prediction]}**")

    prob_df = pd.DataFrame({
        "Clase": class_names,
        "Probabilidad": probabilities
    })

    st.write("Probabilidades por clase:")
    st.dataframe(prob_df)

    # --------------------------------------------------
    # ERROR SI HAY CLASE REAL
    # --------------------------------------------------

    if real_class_option != "No especificar":
        real_class_index = list(class_names).index(real_class_option)

        error = int(prediction != real_class_index)

        if error == 0:
            st.success("✅ Predicción Correcta")
        else:
            st.error("❌ Predicción Incorrecta")

        st.write(f"Error (0=correcto, 1=incorrecto): **{error}**")

# --------------------------------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------------------------------

st.markdown("---")
st.subheader("🔎 Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicción")
plt.ylabel("Real")

st.pyplot(fig)