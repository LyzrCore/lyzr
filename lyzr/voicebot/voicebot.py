from openai import OpenAI
import os
from typing import Optional, Literal
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError


class VoiceBot:  
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_type: Optional[Literal["openai"]] = None,
        model_name: Optional[Literal["tts-1-hd", "tts-1"]] = None,
        model: Optional[LLM] = None,
        voice: Optional[
            Literal["echo", "alloy", "fable", "onyx", "nova", "shimmer"]
        ] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_type = model_type or os.environ.get("MODEL_TYPE") or "openai"
        self.model_name = model_name or os.environ.get("MODEL_NAME") or "tts-1-hd"
        self.voice = voice or os.environ.get("VOICE") or "echo"
        self.model = model or get_model(
            self.api_key, self.model_type, self.model_name, self.voice
        )  # change get_model in lyzr.base.llms to accept **kwargs


    def text_to_speech(
        self,
        text: str,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or self.api_key
        if self.api_key is None:
            raise ValueError("API key must be provided")

        response = self.model.run(input=text)
        # Save the synthesized speech to a file named "tts_output.mp3"
        response.stream_to_file("tts_output.mp3")
        
    def transcribe(self, location):
        # Ensure the model is correctly initialized and matches the required type (e.g., "whisper-1" for transcriptions).
        if not self.model:
            raise ValueError("Model must be provided")

        if self.model.model_type != "openai" or self.model.model_name != "whisper-1":
            raise ValueError("The transcribe function only works with the 'whisper-1' model.")

        # Open the audio file in binary mode and transcribe it using the model.
        with open(location, "rb") as audio_file:
            
            # Use the model's 'run' method, presumably designed to handle different types of tasks.
            response = self.model.run(file=audio_file.read())

            # or you might need to poll for the result. Here, we assume immediate return.
            transcript = response['choices'][0]['text'] if 'choices' in response else ""

            return transcript

    def text_to_notes(self, text):
        if not self.model:
            raise ValueError("Model must be provided")

        if self.model.model_type != "openai" or self.model.model_name != "gpt-3.5-turbo":
            raise ValueError("The text_to_notes function only works with the 'gpt-3.5-turbo' model.")

        # The system message acts as the prompt for the AI.
        system_message = "You are an expert in taking down notes as bullet points and summarizing big conversations. You make sure no detail is left out."

        # Format the user's message that will be sent to the model.
        user_message = f"Here is my conversation: {text}. Can you create bullet-point notes for this?"

        # Use the LLM instance to communicate with OpenAI's API.
        response = self.model.run(
            input=user_message,
            other_messages=[{"role": "system", "content": system_message}]
        )
        # Parse the response to extract the notes.
        notes = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        return notes
    