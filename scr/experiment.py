import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Загрузка датасета Iris
iris = load_iris()
X, y = iris.data, iris.target

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Создание нового эксперимента
mlflow.set_experiment("iris_classification_2")


# Функция для логирования метрик
def log_metrics(y_true, y_pred):
    mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
    mlflow.log_metric("precision", precision_score(y_true,
                                                   y_pred,
                                                   average='weighted'))
    mlflow.log_metric("recall", recall_score(y_true, y_pred,
                                             average='weighted'))
    mlflow.log_metric("f1", f1_score(y_true, y_pred, average='weighted'))


# Эксперимент 1: RandomForestClassifier
with mlflow.start_run(run_name="random_forest"):
    # Задаем параметры
    n_estimators = 20
    max_depth = 5

    # Логируем параметры
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Обучаем модель
    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      random_state=42)
    rf_model.fit(X_train, y_train)

    # Делаем предсказания
    y_pred = rf_model.predict(X_test)

    # Логируем метрики
    log_metrics(y_test, y_pred)

    # Сохраняем модель
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

# Эксперимент 2: SVM
with mlflow.start_run(run_name="svm"):
    # Задаем параметры
    C = 0.2
    kernel = 'rbf'

    # Логируем параметры
    mlflow.log_param("model_type", "SVM")
    mlflow.log_param("C", C)
    mlflow.log_param("kernel", kernel)

    # Обучаем модель
    svm_model = SVC(C=C, kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train)

    # Делаем предсказания
    y_pred = svm_model.predict(X_test)

    # Логируем метрики
    log_metrics(y_test, y_pred)

    # Сохраняем модель
    mlflow.sklearn.log_model(svm_model, "svm_model")

# Эксперимент 3: Логистическая регрессия
with mlflow.start_run(run_name="logistic_regression"):
    # Задаем параметры
    C = 0.1
    max_iter = 100

    # Логируем параметры
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)

    # Обучаем модель
    lr_model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    lr_model.fit(X_train, y_train)

    # Делаем предсказания
    y_pred = lr_model.predict(X_test)

    # Логируем метрики
    log_metrics(y_test, y_pred)

    # Сохраняем модель
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
