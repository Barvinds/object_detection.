import speech_recognition as sr
from googletrans import Translator
import pyttsx3 

recognizer = sr.Recognizer()
engine = pyttsx3.init()
translator = Translator()

def recognize_speech():
    with sr.Microphone() as source: 
        print('Clearing background noise...')
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print('Waiting for message..') 
        audio = recognizer.listen(source, timeout=10)
        print('Done recording..')
        
    try:
        print('Recognizing..')
        result = recognizer.recognize_google(audio, language='en')
        return result
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as ex:
        print("Could not request results; {0}".format(ex))
        return None

def translate_text(text, target_language):
    try:
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as ex:
        print("Translation error:", ex)
        return None

def main():
    source_language = 'en'
    target_language = input('Type the language code you want to translate to (e.g., "fr" for French): ')

    while True:
        print("Speak something...")
        recognized_text = recognize_speech()
        
        if recognized_text:
            print("Recognized:", recognized_text)
            
            translated_text = translate_text(recognized_text, target_language)
            if translated_text:
                print("Translated:", translated_text)
                
                engine.say(translated_text)
                engine.runAndWait()

                # Ask if the user wants to continue translating
                user_input = input("Do you want to continue translating? (yes/no): ").lower()
                if user_input != "yes":
                    break
            else:
                print("Translation failed.")
        else:
            print("Recognition failed.")
            
if __name__ == "__main__":
    main()