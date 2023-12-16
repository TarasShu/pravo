import openai

from openai import OpenAI

from dotenv import load_dotenv
import os
load_dotenv()



client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) #os.getenv('OPENAI_API_KEY')#os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted)

  
class TranslatorOpen:




    def Out_trRtoE(self, content):
        
        engRus = client.chat.completions.create(
            model="gpt-4", #gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "translate from russian to english if it's in english."},
                {"role": "user", "content": content}
            ]
        )
        print("im here")
        return engRus.choices[0].message#.content

    def In_trEtoR(self, content):
      
      
        rusEng = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "translate from russian to english"},
                {"role": "user", "content": content}
            ]
        )
        return rusEng.choices[0].message.content

# Create an instance of the Translator class
translator_instance = TranslatorOpen()

# Example usage
#tr = translator_instance.Out_trRtoE("привет")

