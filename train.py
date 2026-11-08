import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_models():
    print("=" * 60)
    print("Iniciando el entrenamiento del pipeline del Modelo")
    print("=" * 60)

    data_path = 'data/titanic_clean.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo en {data_path}.")

    print(f"Cargando datos desde: {data_path}...")
    df = pd.read_csv(data_path)

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de test: {len(X_test)}")
    print("-" * 50)

    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_cols = ['Sex', 'Embarked']

    print("Definiendo el pipeline...")
    print(f"---> Numérico (StandardScaler): {numerical_cols}")
    print(f"---> Categórico (OneHotEncoder): {categorical_cols}")
    print("-" * 50)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6))
    ])

    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("\nEntrenando Random Forest...")
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print(f"Random Forest Accuracy: {acc_rf:.4f}\n")
    print("Reporte de Clasificación (Random Forest):")
    print(classification_report(y_test, y_pred_rf))
    print("-" * 50)

    print("\nEntrenando Logistic Regression...")
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print(f"Logistic Regression Accuracy: {acc_lr:.4f}\n")
    print("Reporte de Clasificación (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr))
    print("-" * 50)

    print("\nGuardando los modelos...")
    joblib.dump(pipeline_rf, 'model_rf.joblib')
    joblib.dump(pipeline_lr, 'model_lr.joblib')

    print("Modelos guardados exitosamente")
    print("-" * 50)
    print("\nProceso completado exitosamente")


if __name__ == "__main__":
    train_models()







