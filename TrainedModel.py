import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
import seaborn as sns


class TrainedModel:
    def __init__(self, dataset):
        self.beta = 1
        self.cv = KFold(5, shuffle=False)
        # self.basemodel = DecisionTreeClassifier()
        self.basemodel = RandomForestClassifier(n_estimators=20, criterion="entropy")
        self.target = "top_category_1501"
        self.important_target_class = "executive_time"
        self.features = ["top_category_0901", "top_category_1101", "top_category_1301"]
        self.labelencoder = LabelEncoder()
        self.onehotencoder = OneHotEncoder(handle_unknown="ignore")
        self.fit_encoders(dataset)
        self.param_space = {"max_depth": [3, 6, 9, 20, 50]}
        self.model = self.basemodel
        self.best_params = dict()
        self.optimize_params(dataset)

    def performance_metric(self, y_true, y_pred):
        important_target_encode = self.labelencoder.transform(
            [self.important_target_class]
        )
        binary_y_true = (y_true == important_target_encode) * 1
        binary_y_pred = (y_pred == important_target_encode) * 1
        return fbeta_score(binary_y_true, binary_y_pred, beta=self.beta)

    def fit_encoders(self, dataset):
        self.labelencoder.fit(dataset[self.target])
        self.onehotencoder.fit(dataset[self.features])

    def feature_engineering(self, dataset):
        x = self.onehotencoder.transform(dataset[self.features])
        return x

    def feature_engineering_targets(self, dataset):
        y = self.labelencoder.transform(dataset[self.target])
        return y

    def optimize_params(self, dataset):
        gsearch = GridSearchCV(
            estimator=self.basemodel,
            param_grid=self.param_space,
            n_jobs=4,
            scoring=make_scorer(self.performance_metric, greater_is_better=True),
            cv=self.cv,
            verbose=True,
        )
        x_train = self.feature_engineering(dataset)
        y_train = self.feature_engineering_targets(dataset)
        gsearch.fit(x_train, y_train)
        print(
            "Best params found: ",
            gsearch.best_params_,
            "\nwith score: ",
            gsearch.best_score_,
        )
        self.best_params = gsearch.best_params_
        self.model = gsearch.best_estimator_

    def predict(self, dataset, predict_labels=True):
        x_test = self.feature_engineering(dataset)
        y_pred = self.model.predict(x_test)
        if predict_labels:
            return self.labelencoder.inverse_transform(y_pred)
        else:
            return y_pred

    def predict_one(self, top_category_0901, top_category_1101, top_category_1301):
        df = pd.DataFrame(
            [[top_category_0901, top_category_1101, top_category_1301]],
            columns=["top_category_0901", "top_category_1101", "top_category_1301"],
        )
        prediction = self.predict(df)[0]
        return prediction

    def evaluate_performance(self, dataset):
        y_pred_encode = self.predict(dataset, predict_labels=False)
        y_true_encode = self.labelencoder.transform(dataset[self.target])
        performance_score = self.performance_metric(y_true_encode, y_pred_encode)
        print("Performance score (f-beta): ", performance_score)
        return performance_score

    def get_confusion_matrix(self, dataset):
        y_pred = self.predict(dataset, predict_labels=True)
        confmat = confusion_matrix(
            dataset[self.target], y_pred, labels=self.labelencoder.classes_
        )
        return confmat

    def plot_confusion_matrix(self, dataset):
        confmat = self.get_confusion_matrix(dataset)
        cm_df = pd.DataFrame(
            confmat, self.labelencoder.classes_, self.labelencoder.classes_
        )
        cm_df.index.name = "Actual"
        cm_df.columns.name = "Predicted"
        sns.heatmap(cm_df, annot=True)
        plt.show()
