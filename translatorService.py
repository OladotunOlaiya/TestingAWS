import boto3 as boto3
from utilities import Utilities


class TranslatorService:

    def translateText(self):
        translate_client = boto3.client(service_name='translate', use_ssl=True)
        result = translate_client.translate_text(Text=Utilities.get_s3_file_text(self), SourceLanguageCode="en", TargetLanguageCode="fr")
        print('TranslatedText: ' + result.get('TranslatedText'))
        #print('SourceLanguageCode: ' + result.get('SourceLanguageCode'))
        #print('TargetLanguageCode: ' + result.get('TargetLanguageCode'))


