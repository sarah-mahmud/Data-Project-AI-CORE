import yaml
import pandas as pd
from sqlalchemy import create_engine
from eda_tools import DataProcessor, DataAnalyzer


def load_credentials(file_path='credentials.yaml'):
    """Load credentials from a YAML file."""
    with open(file_path, 'r') as file:
        credentials_data = yaml.safe_load(file)
    return credentials_data


class RDSDatabaseConnector:
    """Class for connecting to and interacting with the RDS database."""

    def __init__(self, credentials=None):
        """Initialize the RDSDatabaseConnector with optional credentials."""
        self._credentials = credentials or load_credentials()
        self._engine = None

    def _initialize_engine(self):
        """Initialize the SQLAlchemy engine with the provided credentials."""
        self._engine = create_engine(
            f"postgresql://{self._credentials['RDS_USER']}:{self._credentials['RDS_PASSWORD']}@"
            f"{self._credentials['RDS_HOST']}:{self._credentials['RDS_PORT']}/{self._credentials['RDS_DATABASE']}"
        )

    def extract_loan_payments_data(self, save_to_csv=True, filename='loan_payments.csv'):
        """Extract loan payments data from the RDS database."""
        try:
            self._initialize_engine()
            query = "SELECT * FROM loan_payments"
            df = pd.read_sql(query, self._engine)

            # Save to CSV if specified
            if save_to_csv:
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")

            return df
        except Exception as e:
            print(f"Error extracting loan payments data: {e}")
            return pd.DataFrame()


def load_local_loan_payments_data(filename='loan_payments.csv'):
    """Load loan payments data from a local CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"Shape of the DataFrame: {df.shape}")
        print("Sample of the DataFrame:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return pd.DataFrame()


class DataTransform:
    """Class for transforming data."""

    def __init__(self, df):
        """Initialize with a DataFrame."""
        self.df = df

    def convert_to_numeric(self, column):
        """Convert a column to numeric."""
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

    def convert_to_datetime(self, column):
        """Convert a column to datetime."""
        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')

    def convert_to_categorical(self, column):
        """Convert a column to categorical."""
        self.df[column] = self.df[column].astype('category')


class DataFrameInfo:
    """Class for extracting information from a DataFrame."""

    def __init__(self, df):
        """Initialize with a DataFrame."""
        self.df = df

    def describe_all_columns(self):
        """Describe all columns in the DataFrame."""
        return self.df.describe()

    def extract_statistical_values(self):
        """Extract statistical values from columns and the DataFrame."""
        return self.df.describe().loc[:, ['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    def count_distinct_values(self):
        """Count distinct values in categorical columns."""
        return self.df.nunique()

    def print_dataframe_shape(self):
        """Print out the shape of the DataFrame."""
        print(self.df.shape)

    def generate_null_value_counts(self):
        """Generate count/percentage count of NULL values in each column."""
        return self.df.isnull().sum()


if __name__ == "__main__":
    # Example usage
    rds_connector = RDSDatabaseConnector()
    loan_payments_rds = rds_connector.extract_loan_payments_data()

    local_loan_payments = load_local_loan_payments_data()

    # Perform transformations and analysis as needed
    transformer = DataTransform(local_loan_payments)
    transformer.convert_to_numeric('numeric_column')
    
    info_extractor = DataFrameInfo(local_loan_payments)
    info_extractor.describe_all_columns()

    # Perform additional actions or analyses based on your requirements
