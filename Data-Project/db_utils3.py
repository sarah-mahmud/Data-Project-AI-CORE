import psycopg2
import self as self
import yaml

class RDSDatabaseConnector:
def __init__(self, host, port, database, user, password):
self.host = host
self.port = port
self.database = database
self.user = user
self.password = password
self.connection = None

def connect_to_database(self):
try:
self.connection = psycopg2.connect(
host=self.host,
port=self.port,
database=self.database,
user=self.user,
password=self.password
)
print("Connected to the database.")
except psycopg2.Error as e:
print(f"Error connecting to the database: {e}")

def extract_data(self, query):
if not self.connection:
print("Not connected to the database. Please connect first.")
return None

try:
cursor = self.connection.cursor()
cursor.execute(query)
rows = cursor.fetchall()
return rows
except psycopg2.Error as e:
print(f"Error extracting data from the database: {e}")
return None

def close_connection(self):
try:
if self.connection:
self.connection.close()
print("Connection closed.")
except psycopg2.Error as e:
print(f"Error closing the database connection: {e}")

# Example usage
if __name__ == "__main__":
# Replace with your RDS connection details
rds_connector = RDSDatabaseConnector(
RDS_HOST: eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com
RDS_PASSWORD: EDAloananalyst
RDS_USER: loansanalyst
RDS_DATABASE: payments
RDS_PORT: 5432
)

rds_connector.connect_to_database()

# Replace with your SQL query
query = "SELECT * FROM your_table;"
data = rds_connector.extract_data(query)

if data:
for row in data:
print(row)

rds_connector.close_connection()
