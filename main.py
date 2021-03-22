# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import boto3
from comprehendDetectService import ComprehendDetect
from rekognitionImageService import RekognitionImage
from translatorService import TranslatorService
from utilities import Utilities
from pprint import pprint
import logging


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Rekognition Image Detection')
    print('-' * 88)

    rekognition_client = boto3.client('rekognition')
    street_scene_file_name = ".media/pexels-kaique-rocha-109919.jpg"
    celebrity_file_name = ".media/pexels-pixabay-53370.jpg"
    one_girl_url = 'https://dhei5unw3vrsx.cloudfront.net/images/source3_resized.jpg'
    three_girls_url = 'https://dhei5unw3vrsx.cloudfront.net/images/target3_resized.jpg'
    swimwear_object = boto3.resource('s3').Object('console-sample-images-pdx', 'yoga_swimwear.jpg')
    book_file_name = '.media/pexels-christina-morillo-1181671.jpg'

    street_scene_file_object = boto3.resource('s3').Object('assignment-2212', 'pexels-kaique-rocha-109919.jpg')
    celebrity_file_object = boto3.resource('s3').Object('assignment-2212', 'pexels-pixabay-53370.jpg')

    #Face Recognition

    street_scene_image = RekognitionImage.from_bucket(street_scene_file_object, rekognition_client)
    print(f"Detecting faces in {street_scene_image.image_name}...")
    faces = street_scene_image.detect_faces()
    print(f"Found {len(faces)} faces, here are the first three.")
    for face in faces[:3]:
        pprint(face.to_dict())

    print('*' * 80)
    print(f"Detecting labels in {street_scene_image.image_name}...")
    labels = street_scene_image.detect_labels(100)
    print(f"Found {len(labels)} labels.")
    for label in labels:
        pprint(label.to_dict())
    names = []
    box_sets = []
    colors = ['aqua', 'red', 'white', 'blue', 'yellow', 'green']
    for label in labels:
        if label.instances:
            names.append(label.name)
            box_sets.append([inst['BoundingBox'] for inst in label.instances])

    print('-' * 88)
    print('Translating from English to French')
    print('-' * 88)
    translateService = TranslatorService()
    translateService.translateText()

    print('-' * 88)
    print('Detecting sentiment')
    print('-' * 88)
    client = boto3.client('comprehend')
    comprehendDetectService = ComprehendDetect(client)
    text_in_file = Utilities().get_s3_file_text()

    demo_size = 3
    sentiment = comprehendDetectService.detect_sentiment(text_in_file, 'en')

    print(f"Sentiment: {sentiment['Sentiment']}")
    print("SentimentScore:")
    pprint(sentiment['SentimentScore'])

    print('-' * 88)
    print('Detecting key phrases')
    print('-' * 88)
    phrases = comprehendDetectService.detect_key_phrases(text_in_file, 'en')
    print(f"The first {demo_size} are:")
    pprint(phrases[:demo_size])




