from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.model_train import train_model

def main():
    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

if __name__ == "__main__":
    main()
