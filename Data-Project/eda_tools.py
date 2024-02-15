import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        self.df_before = df_before
        self.df_after = df_after

    def visualize_null_removal(self) -> None:
        """Visualize the removal of NULL values."""
        null_before = (self.df_before.isnull().sum() / len(self.df_before)) * 100
        null_after = (self.df_after.isnull().sum() / len(self.df_after)) * 100

        plt.figure(figsize=(12, 6))
        plt.bar(null_before.index + '_before', null_before, label='Before Imputation', alpha=0.7)
        plt.bar(null_after.index + '_after', null_after, label='After Imputation', alpha=0.7)
        plt.title('Percentage of NULL Values Before and After Imputation')
        plt.xlabel('Columns')
        plt.ylabel('Percentage of NULL Values')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.show()

    def visualize_skewness(self, column: str) -> None:
        """Plot the skewness of a column."""
        plt.figure(figsize=(8, 6))
        plt.hist(self.df_before[column], bins=30, edgecolor='black', alpha=0.7, label='Before Transformation')
        plt.hist(self.df_after[column], bins=30, edgecolor='black', alpha=0.7, label='After Transformation')
        plt.title(f'Skewness Analysis of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    # def visualize_outliers(self, column: str) -> None:
    #     """Plot a boxplot to visualize outliers in a column."""
    #     plt.figure(figsize=(8, 6))
    #     sns.boxplot(x=self.df_before[column], color='skyblue', label='Before Outlier Removal')
    #     sns.boxplot(x=self.df_after[column], color='orange', label='After Outlier Removal')
    #     plt.title(f'Outlier Analysis of {column}')
    #     plt.xlabel(column)
    #     plt.legend()
    #     plt.show()
    
    def visualize_outliers(self, column: str) -> None:
        """Plot a boxplot to visualize outliers in a column."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.df_before[column], color='skyblue')  # Removed 'label' argument
        sns.boxplot(x=self.df_after[column], color='orange')  # Removed 'label' argument
        plt.title(f'Outlier Analysis of {column}')
        plt.xlabel(column)
        plt.show()

    def visualize_correlation_matrix(self, title: str, correlation_matrix: pd.DataFrame) -> None:
        """Visualize the correlation matrix."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5)
        plt.title(title)
        plt.show()

class DataProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df_before = df.copy()
        self.df = df.copy()
        self.df_after = df.copy()

    def _detect_missing_values(self) -> pd.DataFrame:
        """Calculate the number and percentage of missing values in each column."""
        null_counts = self.df.isnull().sum()
        total_values = len(self.df)
        null_percentage = (null_counts / total_values) * 100
        missing_info = pd.DataFrame({'Null Counts': null_counts, 'Percentage Null': null_percentage})
        return missing_info

    def drop_missing_columns(self, threshold: float = 30) -> pd.DataFrame:
        """Drop columns with a percentage of missing values exceeding the threshold."""
        null_info = self._detect_missing_values()
        columns_to_drop = null_info[null_info['Percentage Null'] > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        self.df_after = self.df_after.drop(columns=columns_to_drop)
        return self.df

    def impute_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        """Impute missing values in DataFrame columns."""
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype.kind in 'biufc':
                    if strategy == 'mean':
                        impute_value = self.df[column].mean()
                    elif strategy == 'median':
                        impute_value = self.df[column].median()
                    else:
                        raise ValueError("Invalid imputation strategy. Use 'mean' or 'median'.")
                    
                    self.df[column] = self.df[column].fillna(impute_value)
                    self.df_after[column] = self.df_after[column].fillna(impute_value)
        return self.df
    
    def check_null_after_imputation(self) -> pd.DataFrame:
        """Check for NULL values after imputation."""
        null_info = self._detect_missing_values()
        print(null_info)
        return null_info
    
    # def identify_skewed_columns(self, threshold: float = 0.5) -> pd.Index:
    #     """Identify skewed columns in the data."""
    #     numeric_columns = self.df.select_dtypes(include=np.number).columns
    #     skewness = self.df[numeric_columns].apply(lambda x: x.skew())
    #     skewed_columns = skewness[abs(skewness) > threshold].index
    #     return skewed_columns
    
    def identify_skewed_columns(self, threshold: float = 0.5) -> pd.Index:
        """Identify skewed columns in the data."""
        numeric_columns = self.df.select_dtypes(include=np.number).columns
        skewness = self.df[numeric_columns].apply(lambda x: x.skew())
        skewed_columns = skewness[abs(skewness) > threshold].index
        return skewed_columns

    # def determine_best_transformation(self, skewed_columns: pd.Index) -> dict:
    #     """Determine the transformation that results in the biggest reduction in skew for each identified skewed column."""
    #     transformations = {'log': np.log, 'sqrt': np.sqrt}
    #     best_transformations = {}

    #     for column in skewed_columns:
    #         min_skewness = float('inf')
    #         best_transformation = None

    #         for name, func in transformations.items():
    #             transformed_column = func(self.df[column])
    #             skewness = pd.Series(transformed_column).skew()

    #             if abs(skewness) < min_skewness:
    #                 min_skewness = abs(skewness)
    #                 best_transformation = name

    #         best_transformations[column] = best_transformation

    #     return best_transformations
    
    def determine_best_transformation(self, skewed_columns: pd.Index) -> dict:
        """Determine the transformation that results in the biggest reduction in skew for each identified skewed column."""
        transformations = {'log': lambda x: np.log1p(x), 'sqrt': np.sqrt}
        best_transformations = {}

        for column in skewed_columns:
            min_skewness = float('inf')
            best_transformation = None

            for name, func in transformations.items():
                # Add a small constant value to avoid taking log of zero or negative values
                transformed_column = func(self.df[column] + 1e-10)
                skewness = pd.Series(transformed_column).skew()

                if abs(skewness) < min_skewness:
                    min_skewness = abs(skewness)
                    best_transformation = name

            best_transformations[column] = best_transformation

        return best_transformations

    # def apply_transformations(self, transformation_dict: dict) -> None:
    #     """Apply the identified transformations to the skewed columns."""
    #     transformations = {'log': np.log, 'sqrt': np.sqrt}

    #     for column, transformation in transformation_dict.items():
    #         self.df[column] = transformations[transformation](self.df[column])
    
    # def apply_transformations(self, transformation_dict: dict) -> None:
    #     """Apply the identified transformations to the skewed columns."""
    #     transformations = {'log': np.log, 'sqrt': np.sqrt}

    #     for column, transformation in transformation_dict.items():
    #         if transformation == 'log':
    #             # Add a small constant value to avoid taking log of zero or negative values
    #             self.df[column] = transformations[transformation](self.df[column] + 1e-10)
    #         else:
    #             self.df[column] = transformations[transformation](self.df[column])
    
    def apply_transformations(self, transformation_dict: dict) -> None:
        """Apply the identified transformations to the skewed columns."""
        transformations = {'log': np.log1p, 'sqrt': np.sqrt}

        for column, transformation in transformation_dict.items():
            if transformation == 'log':
                # Apply log transformation only to positive and non-zero values
                positive_values = (self.df[column] > 0) & (~np.isclose(self.df[column], 0))
                if positive_values.any():
                    transformed_values = transformations[transformation](self.df.loc[positive_values, column])
                    # Explicitly cast to a compatible dtype
                    transformed_values = transformed_values.astype(self.df[column].dtype)
                    self.df.loc[positive_values, column] = transformed_values
            elif transformation == 'sqrt':
                # Apply sqrt transformation after taking the absolute value
                transformed_values = transformations[transformation](np.abs(self.df[column]))
                # Explicitly cast to a compatible dtype
                transformed_values = transformed_values.astype(self.df[column].dtype)
                self.df[column] = transformed_values
            else:
                self.df[column] = transformations[transformation](self.df[column])

        # Handle NaN values after transformations
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)


    def save_copy_for_comparison(self, filepath: str) -> None:
        """Save a separate copy of the DataFrame for future comparison."""
        self.df.to_csv(filepath, index=False)

    def identify_outliers(self, columns: pd.Index) -> pd.DataFrame:
        """Identify outliers in the specified columns."""
        outliers = pd.DataFrame()
        for column in columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            outliers = pd.concat([outliers, column_outliers])

        return outliers

    def remove_outliers(self, columns: pd.Index) -> pd.DataFrame:
        """Remove outliers from the specified columns."""
        outliers = self.identify_outliers(columns)
        self.df = self.df.drop(outliers.index)
        return self.df

    # def identify_highly_correlated_columns(self, threshold: float = 0.8) -> pd.Index:
    #     """Identify highly correlated columns based on the correlation threshold."""
    #     correlation_matrix = self.df.corr().abs()
    #     upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    #     highly_correlated_columns = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    #     return highly_correlated_columns
    
    def identify_highly_correlated_columns(self, threshold: float = 0.8) -> pd.Index:
        """Identify highly correlated columns based on the correlation threshold."""
        numeric_columns = self.df.select_dtypes(include=np.number).columns
        correlation_matrix = self.df[numeric_columns].corr().abs()

        # Use np.triu without the need for where
        upper_triangle = correlation_matrix * np.triu(np.ones(correlation_matrix.shape), k=1)

        highly_correlated_columns = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
        return highly_correlated_columns

    def drop_highly_correlated_columns(self, threshold: float = 0.8) -> pd.DataFrame:
        """Drop highly correlated columns based on the correlation threshold."""
        correlated_columns = self.identify_highly_correlated_columns(threshold)
        self.df = self.df.drop(columns=correlated_columns)
        return self.df

    def save_copy_after_correlation_removal(self, filepath: str) -> None:
        """Save a copy of the DataFrame after removing highly correlated columns."""
        self.df.to_csv(filepath, index=False)


if __name__ == "__main__":
    # Example usage:
    
    # Load your DataFrame
    df = pd.read_csv('loan_payments.csv')

    # Data processing before transformation
    processor_before = DataProcessor(df)

    # Visualize null values before imputation
    processor_before.check_null_after_imputation()

    # Remove missing columns
    processor_before.drop_missing_columns()

    # Impute missing values
    processor_before.impute_missing_values()

    # Identify and visualize skewed columns
    skewed_columns = processor_before.identify_skewed_columns()
    analyzer_before = DataAnalyzer(processor_before.df_before, processor_before.df)
    analyzer_before.visualize_skewness(skewed_columns[0])

    # Apply transformations
    transformation_dict = processor_before.determine_best_transformation(skewed_columns)
    processor_before.apply_transformations(transformation_dict)

    # Visualize outliers before removal
    outliers_before = processor_before.identify_outliers(skewed_columns)
    analyzer_before.visualize_outliers(outliers_before.columns[0])

    # Identify and remove highly correlated columns
    highly_correlated_columns_before = processor_before.identify_highly_correlated_columns()
    processor_before.drop_highly_correlated_columns()

    # Save a copy for comparison
    processor_before.save_copy_for_comparison('before_transformations.csv')
    
    # Data processing after transformation
    processor_after = DataProcessor(df)

    # Visualize null values after imputation
    processor_after.check_null_after_imputation()

    # Remove missing columns
    processor_after.drop_missing_columns()

    # Impute missing values
    processor_after.impute_missing_values()

    # Identify and visualize skewed columns
    skewed_columns_after = processor_after.identify_skewed_columns()
    analyzer_after = DataAnalyzer(processor_after.df_before, processor_after.df_after)
    analyzer_after.visualize_skewness(skewed_columns_after[0])

    # Apply transformations
    transformation_dict_after = processor_after.determine_best_transformation(skewed_columns_after)
    processor_after.apply_transformations(transformation_dict_after)

    # Visualize outliers after removal
    outliers_after = processor_after.identify_outliers(skewed_columns_after)
    analyzer_after.visualize_outliers(outliers_after.columns[0])

    # Identify and remove highly correlated columns
    highly_correlated_columns_after = processor_after.identify_highly_correlated_columns()
    processor_after.drop_highly_correlated_columns()

    # Save a copy after transformation and correlation removal
    processor_after.save_copy_after_correlation_removal('after_transformations.csv')

    # Visualize correlation matrix before and after
    correlation_matrix_before = df.corr(numeric_only=True)
    correlation_matrix_after = processor_after.df.corr(numeric_only=True)

    analyzer_before.visualize_correlation_matrix('Correlation Matrix Before Transformation', correlation_matrix_before)
    analyzer_after.visualize_correlation_matrix('Correlation Matrix After Transformation', correlation_matrix_after)

