class VoiceBot:
    def text_to_speech(self, text, model="tts-1-hd", voice="echo", api_key=None):
        # Check if API key is provided
        if api_key is None:
            raise ValueError("API key must be provided")

        # Initialize the OpenAI client with the provided API key
        client = OpenAI(api_key=api_key)

        # Create a speech synthesis request with the provided text, model, and voice
        response = client.audio.speech.create(
            model=model,  # Model for speech synthesis, default is "tts-1-hd". Options are ["tts-1","tts-1-hd"]
            voice=voice,  # Voice for speech synthesis, default is "echo". Options are ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
            input=text,  # Text to be converted to speech
        )

        # Save the synthesized speech to a file named "mainoutput.mp3"
        response.stream_to_file("mainoutput.mp3")

    def transcribe(self, location):
        # Import necessary libraries
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        # Check if CUDA is available and set the device and data type accordingly
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Specify the model ID
        model_id = "distil-whisper/distil-large-v2"

        # Load the model from the Hugging Face model hub
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        # Move the model to the specified device
        model.to(device)

        # Load the processor (tokenizer and feature extractor) from the Hugging Face model hub
        processor = AutoProcessor.from_pretrained(model_id)

        # Create a pipeline for automatic speech recognition (ASR)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Use the pipeline to transcribe the audio file at the specified location
        result = pipe(location)

        # Return the transcribed text
        return result["text"]

    def text_to_notes(self, text):
        # Import necessary libraries
        from openai import OpenAI

        # Initialize the OpenAI client
        client = OpenAI()

        # Create a chat completion request with the specified model and messages
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Model for text generation
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in taking down notes as bullet points and summarizing big conversations. you make sure no detail is left out",
                },
                {
                    "role": "user",
                    "content": f"Here is my conversation: {text}, create notes for this",
                },
            ],
            temperature=1,  # Controls randomness. Higher values (closer to 1) make output more random
            max_tokens=256,  # Maximum number of tokens in the output
            top_p=1,  # Controls diversity via nucleus sampling: 1.0 means "sample from all tokens"
            frequency_penalty=0,  # Controls the penalty for using frequent tokens
            presence_penalty=0,  # Controls the penalty for using new tokens
        )

        # Extract the generated text from the response
        text = response.choices[0].message.content

        # Return the generated text
        return text

    def record_audio(self):
        # Import necessary libraries
        import sounddevice as sd
        import numpy as np
        import scipy.io.wavfile as wav
        import queue

        # Set the sample rate and number of channels
        sample_rate = 44100
        channels = 2

        # Define a callback function to be called for each block of audio data
        def callback(indata, frames, time, status):
            # If there is an error, print it
            if status:
                print(status)
            # Add the incoming data to the queue
            q.put(indata.copy())

        # Create a queue to hold the incoming audio data
        q = queue.Queue()

        # Create an input stream with the specified sample rate, number of channels, and callback function
        stream = sd.InputStream(
            callback=callback, channels=channels, samplerate=sample_rate
        )

        # Start the input stream
        stream.start()

        # Wait for the user to press Enter to stop the recording
        input("Press Enter to stop the recording...")

        # Stop the input stream
        stream.stop()

        # Concatenate all the audio data in the queue
        recording = np.concatenate(list(q.queue))

        # Write the audio data to a WAV file
        wav.write("output.wav", sample_rate, recording)
