from googletrans import Translator

def translate_hindi_to_english(text):
    translator = Translator()
    result = translator.translate(text, src='hi', dest='en')
    return result.text 


print(translate_hindi_to_english("mujhe return request dalni hai"))