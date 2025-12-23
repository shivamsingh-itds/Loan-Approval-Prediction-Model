import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

def train_model(X_train, X_test, y_train, y_test, preprocessor):

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save Model
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(pipeline, model_dir / "loan_model.pkl")

    return pipeline
