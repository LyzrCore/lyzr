from openai import OpenAI
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import queue

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
        
        # Save the synthesized speech to a file named "mainoutput.mp3"
        response.stream_to_file("mainoutput.mp3")

    def transcribe(self, location):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/distil-large-v2"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
        result = pipe(location)
        return result['text']
    
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
    
    def record_audio(self):
        sample_rate = 44100
        channels = 2

        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())

        q = queue.Queue()
        stream = sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate)
        stream.start()
        input("Press Enter to stop the recording...")
        stream.stop()
        recording = np.concatenate(list(q.queue))
        wav.write('recording.wav', sample_rate, recording)

       
    def record_and_summarize(self):
        sample_rate = 44100
        channels = 2
        def callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(indata.copy())
        q = queue.Queue()
        stream = sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate)
        stream.start()
        input("Press Enter to stop the recording...")
        stream.stop()
        recording = np.concatenate(list(q.queue))
        wav.write('output.wav', sample_rate, recording)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "distil-whisper/distil-large-v2"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        result = pipe('output.wav')
        
        text = result['text']
        
        
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
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        summarized_text = response.choices[0].message.content
        return summarized_text
        