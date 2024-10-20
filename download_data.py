import boto3

# Initialize the S3 client
s3 = boto3.client('s3')

# Define your bucket and file names
bucket_name = 'electricvehiclebyyash'
file_name = 'EV_Dataset.csv'
local_path = '/home/ubuntu/EV_Dataset.csv'  # Path to save the dataset

try:
    # Download the file from S3
    s3.download_file(bucket_name, file_name, local_path)
    print(f"Dataset downloaded successfully to {local_path}")
except Exception as e:
    print(f"Error downloading the dataset: {e}")
