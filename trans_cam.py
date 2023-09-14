import cv2
import pytesseract
from googletrans import Translator
import pyttsx3 

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Initialize the OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

translator = Translator()
engine = pyttsx3.init()



def recognize_text(frame):
    text = pytesseract.image_to_string(frame)
    return text



def translate_text(text, target_language):
    try:
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as ex:
        print("Translation error:", ex)
        return None

def main():
    target_language = input('Type the language code you want to translate to (e.g., "fr" for French): ')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break
        
        
        
        recognized_text = recognize_text(frame)
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
            
            
            cv2.imshow('Webcam Feed', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()