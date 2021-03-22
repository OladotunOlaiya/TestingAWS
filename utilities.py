import boto3

class Utilities:

    def get_s3_file_text(self):
        s3_client = boto3.client('s3')  # low-level functional API
        s3_resource = boto3.resource('s3')  # high-level object-oriented API
        s3_obj = s3_client.get_object(Bucket='assignment-2212', Key='sample-file.txt')
        return s3_obj['Body'].read().decode("utf-8")
