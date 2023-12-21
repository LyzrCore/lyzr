from openai import OpenAI

class VoiceBot():

    def text_to_speech(self, text, model="tts-1-hd", voice="echo", api_key=None):
        if api_key is None:
            raise ValueError("API key must be provided")
        
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model, #["tts-1","tts-1-hd"]
            voice=voice, #['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
            input=text,  
        )
        
        # Save the synthesized speech to a file named "tts_output.mp3"
        response.stream_to_file("tts_output.mp3")

        
    def transcribe(self, location):
        client = OpenAI()
        
        with open(location, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            
        return transcript.text
    
    def text_to_notes(self, text):
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in taking down notes as bullet points and summarizing big conversations. you make sure no detail is left out"
                },
                {
                    "role": "user",
                    "content": f"Here is my conversation: {text}, create notes for this"
                }
            ],
            temperature=1,  
            max_tokens=800, 
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0  
        )
        notes = response.choices[0].message.content
        return notes
    
    
