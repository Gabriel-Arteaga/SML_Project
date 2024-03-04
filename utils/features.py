
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    Normalizer
)


class FeatureEngineering:
    """
    Collection of methods to engineer additional features and
    add to the original dataframe. Note that some methods can
    require previous steps to have run first.
    """

    def __init__(self):
        pass

    @staticmethod
    def numeric_class_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a column with numeric class values.

        Args:
            df: Dataframe with data about actors and movies

        Returns:
            df: Dataframe with column "Lead numeric" appended
        """

        df["Lead"] = df["Lead"].astype("category")
        df["Lead numeric"] = df["Lead"].cat.codes

        return df

    @staticmethod
    def encode_class_column(df: pd.DataFrame):
        """
        Encode column lead to numerical values.

        Args:
            df: Dataframe with data about actors and movies.

        Returns:
            Null: Encodes "Lead" column with numerical values.
        """
        # Encode Lead to numeric values
        df["Lead"] = df["Lead"].astype("category").cat.codes

    @staticmethod
    def calculate_total_actors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total number of actors.
        """

        # Total number of actors of both genders
        df["Total actors"] = df["Number of male actors"] + df["Number of female actors"]

        return df

    def calculate_relative_shares(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative shares for relevant columns and append to
        the original dataframe. Calculation of word shares is
        dependent on first calling calculate_totals().

        Args:
            df: Dataframe with data about actors and movies

        Returns:
            df: Dataframe with the following columns appended:
            "Female word share",
            "Male word share", "Female actor share", "Male actor share"
        """

        # Calculate totals first if needed
        if "Total actors" not in df.columns:
            df = self.calculate_total_actors(df)

        # Calculate gender word shares
        # This excludes the lead character to avoid leakage in training
        df["Female word share"] = (
                df["Number words female"] / df["Total words"]
        )
        df["Male word share"] = 1 - df["Female word share"]

        df["Lead word share"] = (
                df["Number of words lead"] / df["Total words"]
        )

        # Calculate gender actor shares
        df["Total actors"] = (
                df["Number of female actors"] + df["Number of male actors"]
        )
        df["Female actor share"] = (
                df["Number of female actors"] / df["Total actors"]
        )
        df["Male actor share"] = 1 - df["Female actor share"]

        return df

    @staticmethod
    def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:

        df["Female / male actor ratio"] = (
                df["Number of female actors"] / df["Number of male actors"]
        )
        df["Lead word ratio"] = (
                df["Number of words lead"] / df["Total words"])

        df = df.replace([np.inf, -np.inf, np.nan], 0)

        return df

    @staticmethod
    def log_features(df: pd.DataFrame) -> pd.DataFrame:

        log_columns = [
            "Total words",
            "Number words female",
            "Number words male",
            "Number of words lead",
            "Difference in words lead and co-lead",
            "Gross",
        ]

        for column in log_columns:
            df[column] = np.log(df[column]).replace([np.inf, -np.inf], 0)

        return df

    @staticmethod
    def decade(df: pd.DataFrame) -> pd.DataFrame:

        df["Decade"] = df["Year"].astype(str).str[:3] + "0s"
        df["Decade"] = df["Decade"].astype("category").cat.codes


        return df

    def gross_decade_ratio(self, df: pd.DataFrame) -> pd.DataFrame:

        if "Decade" not in df.columns:
            df = self.decade(df)

        df_avg_gross_decade = pd.DataFrame(
            df.groupby("Decade")["Gross"]
                .mean()
                .reset_index()
                .rename(columns={"Gross": "Avg gross decade"})
        )

        df = df.join(df_avg_gross_decade.set_index("Decade"), on="Decade")
        df["Gross ratio vs decade"] = df["Gross"] / df["Avg gross decade"]

        return df


    @staticmethod
    def calculate_difference(
            df: pd.DataFrame, column_b: str, column_a: str
            ) -> pd.Series:
        """
        Calculates difference between columnB and columnA,
        i.e. columnB-columnA.
        Args:
            df: Dataframe with data about actors and movies
            column_b: String name of column that will become subject of
            subtraction.
            column_a: String name of column which will be term to subtract
            column_b by.
        Returns:
            Series: Series with one column contatining the difference
            of the two input columns.
        """
        # Check if column B exists in df
        assert column_b in df.columns, "column_b does not exist in df!"

        # Check if column A exists in df
        assert column_a in df.columns, "columnA does not exist in df!"

        # Check if columnB contains numerical values
        assert (
            df[column_b].dtypes == np.int64 or df[column_b].dtypes == np.float64
        ), "columnB does not contain a numerical value!"

        # Check if columnA contains numerical values
        assert (
            df[column_a].dtypes == np.int64 or df[column_a].dtypes == np.float64
        ), "column_a does not contain a numerical value!"

        # Calculates the difference of column B and A
        result = df[column_b] - df[column_a]

        return result

    @staticmethod
    def calculate_mean_yearly(
        df: pd.DataFrame, columns: list | str
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates yearly mean of the desired feature.
        Args:
            df: Dataframe with data about actors and movies
            columns: String name of column or list of string name of columns
            of which the mean will be calculated.
        Returns:
            If type(columns) == list:
                DataFrame: A DataFrame with several columns containing
                those feature's yearly mean
            else:
                Series: Series with one column containing desired feature's
                yearly mean.
        """
        # Calculate the yearly value of column
        if isinstance(columns, list):
            # Check if the columns exist within the DataFrame
            assert all(
                column in df.columns.tolist() for column in columns
            ), "One or several columns does not exist in df!"

            # Check if columns contains numerical value
            assert all(
                [dtype in (np.int64, np.float64) for dtype in df[columns].dtypes.tolist()]
            ), "One or several columns does not contatin numerical values!"

            df_columns_by_year = pd.DataFrame(df.groupby("Year")[columns].mean())
            return df_columns_by_year

        else:
            # Check if column exists in df
            assert columns in df.columns, "column does not exist in df!"

            # Check if column contains numerical values
            assert (
                df[columns].dtypes == np.int64 or df[columns].dtypes == np.float64
            ), "column does not contain a numerical value!"
            # Calculate the yearly value of column
            total_yearly = df.groupby("Year")[columns].sum()

            # Count total movies by year
            movies_yearly = df.Year.value_counts()

            # Calculate yearly mean of column for each row in df
            mean_list = []
            for index, value in df["Year"].iteritems():
                mean_list.append(total_yearly[value] / movies_yearly[value])

            return pd.Series(mean_list)

    @staticmethod
    def scaling(df: pd.DataFrame, method: str, columns: list | str = None):
        """
        A function which scales an entire dataframe or specified features in the dataframe.

        Methods:

        MaxAbs Scaling: This method scales and translates each feature
        individually such that the maximal absolute value of each feature
        in the training set will be 1.0. It does not shift/center the data,
        and thus does not destroy any sparsity.

        MinMax Scaling: Rescaling of all values in a feature in the range 0 to 1.
        The min value in the original range will take the value 0,
        the max value will take 1 and the rest of the values in between the two will be appropriately scaled.

        Robust Scaling: Scale features using statistics that are robust to outliers.
        This Scaler removes the median and scales the data according to the quantile range (Interquartile range).
        The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

        Standard Scaling: Standardize features by removing the mean and scaling to unit variance.

        Normalize scaling: [To add description]
        Args:
            df: Dataframe with data about actors and movies
            Method: String name of scaling method, either MaxAbs, MinMax, Robust or Standard.
            columns: list or str of features to be scaled, default is None,
                     if no argument is passed the entire dataframe will be scaled.
        Returns:
            void: Changes the desired dataframe internally, returns nothing.

        """
        assert method.lower() in [
            "maxabs",
            "minmax",
            "robust",
            "standard",
            "normalize",
        ], "Method should contain a string with name of scaling method, either MaxAbs, MinMax, Robust, Standard or Normalize"

        # Check if using assigning specific columns
        if columns != None:
            # Check if columns is a list
            if isinstance(columns, list):
                # Check if the columns exist within the DataFrame
                assert all(
                    column in df.columns.tolist() for column in columns
                ), "One or several columns does not exist in df!"

                # Check if columns contains numerical value
                assert all(
                    [
                        dtype in (np.int64, np.float64)
                        for dtype in df[columns].dtypes.tolist()
                    ]
                ), "One or several columns does not contatin numerical values!"

                # Transform features to fit scaler
                features = [np.array(df[feature]).reshape(-1, 1) for feature in columns]

            else:
                # Check if column exists
                assert columns in df.columns, "column does not exist in df!"

                # Check if numerical value
                assert (
                    df[columns].dtypes == np.int64 or df[columns].dtypes == np.float64
                ), "column does not contain a numerical value!"

                # Transform features
                features = [np.array(df[column]).reshape(-1, 1)]

                # Create a list containing column
                columns = list(columns)

        else:
            if "Lead" in df.columns.tolist():
                # Remove lead from features and reshape all features
                features = [
                    np.array(df[feature]).reshape(-1, 1)
                    for feature in df.columns.drop("Lead")
                ]
                columns = df.columns.drop("Lead")

            else:
                # Reshape all features to fit scaler
                features = [
                    np.array(df[feature]).reshape(-1, 1) for feature in df.columns
                ]

                # Create a local variable with columns
                columns = df.columns

        for feature, column_name in zip(features, columns):
            # If using MaxAbsScaling method
            if method.lower() == "maxabs":
                df[column_name] = MaxAbsScaler().fit_transform(feature)

            # If using MinMaxScaling method
            elif method.lower() == "minmax":
                df[column_name] = MinMaxScaler().fit_transform(feature)

            # If using RobustScaling method
            elif method.lower() == "robust":
                df[column_name] = RobustScaler().fit_transform(feature)

            # If using StandarScaling method
            elif method.lower() == "standard":
                try:
                    df[column_name] = StandardScaler().fit_transform(feature)
                except ValueError:
                    print("Column name:", column_name, "\nFeature:", feature)

            # If using normalizer method
            elif method.lower() == "normalize":
                df[column_name] = Normalizer().fit_transform(feature)

    def run_feature_engineering(
        self,
        df: pd.DataFrame,
        scaling_method: str = "standard",
        scaling: boolean = True,
        add_numeric_class_column: boolean = False,
        encode_class: boolean = True,
        total_actors: boolean = True,
        relative_shares: boolean = True,
        ratios: boolean = True,
        decade: boolean = True,
        decade_gross_ratio: bool = True,
        differences: boolean = True,
        abs_diff: boolean = False,
        yearly_mean: boolean = True,
        yearly_mean_diff: boolean = True,
    ) -> pd.DataFrame:
        """Run desired methods in this class in sequence
        Args:
            df: A dataframe with data about actors and movies.
            scaling_method: A string with scaling method one out of "standard",
            "minmax", "maxabs", "robust" or "normalize"
            scaling: boolean, if True scales data with desired scaling_method
            add_numeric_class_column: boolean, if True adds a new numeric 'Lead' column
            encode_class: boolean, if True encodes existing 'Lead' column to numeric
            total_actors: boolean, if True adds column with total actors
            relative_shares: boolean, if True, adds columns with relative shares
            ratios: boolean, if True, adds columns with ratios
            decade: boolean, if True adds categorical decade and numeric decade column
            differences: boolean, if True adds differences of some columns
            abs_diff: boolean, if True adds columns of the absolute value from difference
            taken of 'Age Lead' & 'Age Co-Lead' and 'Mean Age Male' & 'Mean Age Female'.
            yearly_mean: boolean, if True adds columns with the yearly mean of the original
            features.
            yearly_mean_diff: boolean, if True takes the difference of yearly mean and the
            original features.
        Returns:
                df: A dataframe with data about actors and movies.
        """

        # Create numeric class column
        if add_numeric_class_column:
            self.numeric_class_column(df)

        # Encode 'Lead' column to numeric values
        if encode_class:
            self.encode_class_column(df)

        # Calculate new aggregated totals columns
        if total_actors:
            self.calculate_total_actors(df)

        # Calculate relative shares between some columns
        if relative_shares:
            self.calculate_relative_shares(df)

        # Add ratios between features
        if ratios:
            self.calculate_ratios(df)

        # Create decade feature
        if decade:
            self.decade(df)

        if decade_gross_ratio:
            self.gross_decade_ratio(df)

        # Calculate differences between some columns
        if differences and abs_diff:
            # Calculate difference in words by gender
            df["Difference Words Gender"] = self.calculate_difference(
                df, "Number words male", "Number words female"
            )

            # Calculdate difference of actors by gender
            df["Difference Actors"] = self.calculate_difference(
                df, "Number of male actors", "Number of female actors"
            )

            df["Difference Age Lead"] = self.calculate_difference(
                df, "Age Lead", "Age Co-Lead"
            )
            df["Difference Age Lead Abs"] = abs(
                self.calculate_difference(df, "Age Lead", "Age Co-Lead")
            )
            df["Difference Mean Age"] = self.calculate_difference(
                df, "Mean Age Male", "Mean Age Female"
            )
            df["Difference Mean Age Abs"] = abs(
                self.calculate_difference(df, "Mean Age Male", "Mean Age Female")
            )

        elif differences == True and abs_diff == False:
            # Calculate difference in words by gender
            df["Difference Words Gender"] = self.calculate_difference(
                df, "Number words male", "Number words female"
            )

            # Calculdate difference of actors by gender
            df["Difference Actors"] = self.calculate_difference(
                df, "Number of male actors", "Number of female actors"
            )

            # Calculate age lead difference
            df["Difference Age Lead"] = self.calculate_difference(
                df, "Age Lead", "Age Co-Lead"
            )

            # Calculate mean age difference
            df["Difference Mean Age"] = self.calculate_difference(
                df, "Mean Age Male", "Mean Age Female"
            )

        if yearly_mean and yearly_mean_diff:
            original_features = [
                "Number words female",
                "Total words",
                "Number of words lead",
                "Difference in words lead and co-lead",
                "Number of male actors",
                "Number of female actors",
                "Number words male",
                "Gross",
                "Mean Age Male",
                "Mean Age Female",
                "Age Lead",
                "Age Co-Lead",
            ]
            mean_yearly = [
                "Yearly mean Number words female",
                "Yearly mean Total words",
                "Yearly mean Number of words lead",
                "Yearly mean Difference in words lead and co-lead",
                "Yearly mean Number of male actors",
                "Yearly mean Number of female actors",
                "Yearly mean Number words male",
                "Yearly mean Gross",
                "Yearly mean Mean Age Male",
                "Yearly mean Mean Age Female",
                "Yearly mean Age Lead",
                "Yearly mean Age Co-Lead",
            ]
            mean_diff_yearly = [
                "Yearly mean diff Number words female",
                "Yearly mean diff Total words",
                "Yearly mean diff Number of words lead",
                "Yearly mean diff Difference in words lead and co-lead",
                "Yearly mean diff Number of male actors",
                "Yearly mean diff Number of female actors",
                "Yearly mean diff Number words male",
                "Yearly mean diff Gross",
                "Yearly mean diff Mean Age Male",
                "Yearly mean diff Mean Age Female",
                "Yearly mean diff Age Lead",
                "Yearly mean diff Age Co-Lead",
            ]
            # Calculate the yearly mean for all features and adds them to dataframe
            for orig, name in zip(original_features, mean_yearly):
                df[name] = self.calculate_mean_yearly(df, orig)

            # Take the difference of the yearly mean and the original feature
            for orig, yearl, name in zip(
                original_features, mean_yearly, mean_diff_yearly
            ):
                df[name] = self.calculate_difference(df, yearl, orig)

        elif yearly_mean == True and yearly_mean_diff == False:
            original_features = [
                "Number words female",
                "Total words",
                "Number of words lead",
                "Difference in words lead and co-lead",
                "Number of male actors",
                "Number of female actors",
                "Number words male",
                "Gross",
                "Mean Age Male",
                "Mean Age Female",
                "Age Lead",
                "Age Co-Lead",
            ]
            mean_yearly = [
                "Yearly mean Number words female",
                "Yearly mean Total words",
                "Yearly mean Number of words lead",
                "Yearly mean Difference in words lead and co-lead",
                "Yearly mean Number of male actors",
                "Yearly mean Number of female actors",
                "Yearly mean Number words male",
                "Yearly mean Gross",
                "Yearly mean Mean Age Male",
                "Yearly mean Mean Age Female",
                "Yearly mean Age Lead",
                "Yearly mean Age Co-Lead",
            ]

            # Calculate the yearly mean for all features and adds them to dataframe
            for orig, name in zip(original_features, mean_yearly):
                df[name] = self.calculate_mean_yearly(df, orig)

        # Perform scaling
        if scaling:
                self.scaling(df, scaling_method)

        return df
