# Predict President Activities

Example:
```
dataset = pd.read_csv("train_df.csv").fillna("none")
dataset_A, dataset_B = train_test_split(dataset, test_size=0.2, shuffle=False)

trained_model_A = TrainedModel(dataset_A)
performance = trained_model_A.evaluate_performance(dataset_B)
confusion_matrix = trained_model_A.get_confusion_matrix(dataset_B)
```
