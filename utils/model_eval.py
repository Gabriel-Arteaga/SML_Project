import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import (
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

class ModelEvaluation:
    def __init__(self):
        pass

    def cross_val(
        self,
        estimator: object,
        features: pd.DataFrame,
        target: pd.Series,
        nb_folds: int = 5,
        use_preprocess=False,
        preprocess_option: str = "standard",
        use_XGB = False,
        stopping_rounds: int = 20,
    ):
        """
        Perform cross-validation and print model performance report.
        Args:
            estimator: the estimator used for fitting and testing
            features: set of features to train the estimator on
            target: labels in the classification
            nb_folds: number of folds (test sets) to use in the cross-validation
            preprocess_option: what scaling / standardization method to use
        """

        if type(features) == pd.DataFrame:
            feature_columns = list(features.columns)

        elif type(features) == pd.Series:
            feature_columns = features.name
            features = np.array(features).reshape(-1, 1)

        # Perform scaling of features
        if use_preprocess:
            features = self.pre_process(features, preprocess_option)

        # Create k stratified folds
        k_folds = StratifiedKFold(n_splits=nb_folds)

        # Scoring metrics to be used
        scoring = {
            "overall_acc": [],
            "female_acc": [],
            "male_acc": [],
            "train_acc": [],
            "female_FP": [],
            "female_FN": [],
            "male_FP": [],
            "male_FN": [],
        }

        test_sets = pd.DataFrame()
        for train_idx, test_idx in k_folds.split(features, target):
            x_train, y_train = features.iloc[train_idx], target.iloc[train_idx]
            x_test, y_test = features.iloc[test_idx], target.iloc[test_idx]
            test_sets = pd.concat([test_sets, pd.DataFrame(x_test)], axis="rows")

            # Fit model and do prediction
            if use_XGB:
                model = estimator.fit(x_train, y_train,eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds = stopping_rounds)
            else:
                model = estimator.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_train = model.predict(x_train)

            # Put results in a dataframe
            df_res = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})

            # Calculate and append metrics for this fold; 1 is male and 0 is female
            overall_acc = np.mean(y_pred == y_test)
            train_acc = np.mean(y_pred_train == y_train)

            female_acc = (
                df_res.loc[
                    (df_res["y_test"] == 0) & (df_res["y_pred"] == df_res["y_test"])
                ].count()
                / df_res.loc[(df_res["y_test"] == 0)].count()
            )

            male_acc = (
                df_res.loc[
                    (df_res["y_test"] == 1) & (df_res["y_pred"] == df_res["y_test"])
                ].count()
                / df_res.loc[(df_res["y_test"] == 1)].count()
            )

            scoring["overall_acc"].append(overall_acc)
            scoring["female_acc"].append(female_acc)
            scoring["male_acc"].append(male_acc)
            scoring["train_acc"].append(train_acc)

        # Calculate key metrics for output
        overall_acc = scoring["overall_acc"]
        female_acc = scoring["female_acc"]
        male_acc = scoring["male_acc"]
        train_acc = scoring["train_acc"]

        # Summary key metrics on test folds
        avg_acc, min_acc, max_acc = (
            np.mean(overall_acc),
            np.min(overall_acc),
            np.max(overall_acc),
        )
        avg_female_acc, min_female_acc, max_female_acc = (
            np.mean(female_acc),
            np.min(female_acc),
            np.max(female_acc),
        )
        avg_male_acc, min_male_acc, max_male_acc = (
            np.mean(male_acc),
            np.min(male_acc),
            np.max(male_acc),
        )
        avg_train_acc, min_train_acc, max_train_acc = (
            np.mean(train_acc),
            np.min(train_acc),
            np.max(train_acc),
        )

        # Confidence interval using Hoeffding's inequality
        lower_bound, upper_bound, conf_level = self.confidence_interval(
            features, nb_folds, avg_acc
        )

        print("----------- Cross-validation report -----------\n")
        print(f"Model: {estimator}\n")
        print(f"Feature set: {feature_columns}\n")
        print(f"Number of folds: {nb_folds}\n")
        print("Performance:")
        print(
            f"- Accuracy: {avg_acc:.3f} (avg), {min_acc:.3f} (min), {max_acc:.3f} (max)"
        )
        print(
            f"- Accuracy, {conf_level * 100} % confidence interval: {lower_bound:.3f}-{upper_bound:.3f}"
        )
        print(
            f"- Accuracy, female: {avg_female_acc:.3f} (avg), {min_female_acc:.3f} (min), {max_female_acc:.3f} (max)"
        )
        print(
            f"- Accuracy, male: {avg_male_acc:.3f} (avg), {min_male_acc:.3f} (min), {max_male_acc:.3f} (max)"
        )
        print(
            f"- Training accuracy: {avg_train_acc:.3f} (avg), {min_train_acc:.3f} (min), {max_train_acc:.3f} (max)"
        )
        print("---------------------------------------------\n")

    @staticmethod
    def confidence_interval(features, nb_folds, avg_acc):

        conf_level = 0.95
        alpha = 1 - conf_level
        n = features.shape[0] / nb_folds
        delta = (1 / np.sqrt(n)) * np.sqrt((1 / 2) * np.log(2 / alpha))
        lower_bound = avg_acc - delta
        upper_bound = avg_acc + delta

        return lower_bound, upper_bound, conf_level

    @staticmethod
    def shuffle_test(
        estimator: object,
        features: pd.DataFrame,
        target: pd.Series,
        nb_folds: int,
    ):
        """
        Perform column-wise shuffle testing. This means that one column at a time is
        randomly shuffled, and the test performance of this model is compared to a
        baseline accuracy where all features (non-shuffled) are included.
        Args:
            estimator: the estimator used for fitting and testing
            features: set of features to train the estimator on
            target: labels in the classification
            nb_folds: number of folds (test sets) to use in the cross-validation
        """

        # Store original features
        original_features = features

        # Do baseline crossval with all features
        cv_output = cross_validate(
            estimator,
            features,
            target,
            cv=nb_folds,
            scoring=["accuracy"],
        )

        # Calculate baseline accuracy
        accuracy = cv_output["test_accuracy"]
        avg_acc, min_acc, max_acc = (
            np.mean(accuracy),
            np.min(accuracy),
            np.max(accuracy),
        )

        shuffle_dict = {}

        # Shuffle one column at a time
        for column in original_features.columns:
            to_shuffle = features[column]
            shuffled = to_shuffle.sample(frac=1).values
            features = features.drop([column], axis="columns")
            features[column] = shuffled

            cv_output = cross_validate(
                estimator,
                features,
                target,
                cv=nb_folds,
                scoring=["accuracy"],
            )

            # Calculate accuracy with shuffled feature
            shuffle_accuracy = cv_output["test_accuracy"]
            shuffle_avg_acc, shuffle_min_acc, shuffle_max_acc = (
                np.mean(shuffle_accuracy),
                np.min(shuffle_accuracy),
                np.max(shuffle_accuracy),
            )

            # calculate difference vs baseline
            avg_acc_diff, avg_max_diff, avg_min_diff = (
                shuffle_avg_acc / avg_acc - 1,
                shuffle_max_acc / max_acc - 1,
                shuffle_min_acc / min_acc - 1,
            )

            shuffle_dict[
                column
            ] = f"Avg: {avg_acc_diff:.4f}, Max: {avg_max_diff:.4f}, Min: {avg_min_diff:.4f}"

        print("----------- Shuffle testing report -----------\n")
        print("Relative change in accuracy (average, max, min)")
        print("(in %, not % points)\n")
        for key in shuffle_dict.keys():
            print(key)
            print(shuffle_dict[key])
            print("\n")
