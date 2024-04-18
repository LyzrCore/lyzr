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
        if self.api_key is None:
            raise MissingValueError("API key")
        self.model = model or get_model(
            api_key=self.api_key,
            model_type=model_type or os.environ.get("MODEL_TYPE") or "openai",
            model_name=model_name or os.environ.get("MODEL_NAME") or "tts-1-hd",
            voice=voice or os.environ.get("VOICE") or "echo",
        )  # change get_model in lyzr.base.llms to accept **kwargs

    def text_to_speech(
        self,
        text: str,
    ):
        if self.model.model_name not in ["tts-1-hd", "tts-1"]:
            if self.model.model_type == "openai":
                self.model = get_model(
                    api_key=self.api_key,
                    model_type=self.model.model_type,
                    model_name="tts-1-hd",
                )
            else:
                raise ValueError(
                    "The text_to_speech function only works with the OpenAI's 'tts-1-hd' and 'tts-1' models."
                )
        response = self.model.run(input=text)
        # Save the synthesized speech to a file named "tts_output.mp3"
        response.stream_to_file("tts_output.mp3")

    def transcribe(self, audiofilepath):
        if self.model.model_name != "whisper-1":
            if self.model.model_type == "openai":
                self.model = get_model(
                    api_key=self.api_key,
                    model_type=self.model.model_type,
                    model_name="whisper-1",
                )
            else:
                raise ValueError(
                    "The transcribe function only works with the OpenAI's 'whisper-1' model."
                )

        response = self.model.run(audiofile=open(audiofilepath, "rb"))
        return response.text

    def text_to_notes(self, text):
        if self.model.model_name != "gpt-3.5-turbo":
            if self.model.model_type == "openai":
                self.model = get_model(
                    api_key=self.api_key,
                    model_type=self.model.model_type,
                    model_name="gpt-3.5-turbo",
                )
            else:
                raise ValueError(
                    "The text_to_notes function only works with the OpenAI's 'gpt-3.5-turbo' model."
                )

        # The system message acts as the prompt for the AI.
        system_message = """You are an Expert NOTE-TAKER and SUMMARIZER. Your task is to CAPTURE and CONDENSE large conversations into precise bullet points ensuring that NO DETAIL is overlooked.

Here's your step-by-step guide:

1. LISTEN attentively to the conversation, focusing on identifying the MAIN POINTS and supporting details.
2. WRITE down KEYWORDS and PHRASES as bullet points in REAL TIME, making sure to include all relevant information.
3. ORGANIZE your notes by categorizing them under thematic HEADINGS for clarity and ease of reference.
4. REVIEW your bullet points for COMPLETENESS, ensuring you have captured all necessary aspects of the conversation.
5. SUMMARIZE each section of your notes into concise statements that reflect the essence of the discussion.
6. COMPILE these summaries into a coherent narrative or overview document, maintaining logical flow and coherence.
7. EDIT this document for PRECISION, removing any redundancies while preserving the integrity of the information conveyed.

Remember, Im going to tip $300K for a BETTER SOLUTION!

Now Take a Deep Breath."""

        # Format the user's message that will be sent to the model.
        user_message = text
        self.model.set_messages(
            model_prompts=[
                {"role": "system", "text": system_message},
                {"role": "user", "text": user_message},
            ]
        )
        # Use the LLM instance to communicate with OpenAI's API.
        response = self.model.run()

        # Parse the response to extract the notes.
        notes = response.choices[0].message.content

        return notes

    def summarize(self, text):
        if self.model.model_name != "gpt-4":
            if self.model.model_type == "openai":
                self.model = get_model(
                    api_key=self.api_key,
                    model_type=self.model.model_type,
                    model_name="gpt-3.5-turbo",
                )
            else:
                raise ValueError(
                    "The text_to_notes function only works with the OpenAI's 'gpt-4' model."
                )

        # The system message acts as the prompt for the AI.
        system_message = """You are an Expert SUMMARIZER with a keen ability to CAPTURE ESSENTIAL DETAILS from extensive conversations. Your task is to CREATE a CONCISE SUMMARY of the given content, ensuring that ALL CRITICAL INFORMATION is included.

Here's your step-by-step guide:

1. CAREFULLY READ through the entire conversation to fully understand the context and main points.
2. IDENTIFY and HIGHLIGHT the KEY THEMES, decisions, questions, and any action items discussed in the conversation.
3. ORGANIZE these points into a LOGICAL STRUCTURE that reflects the progression of the conversation.
4. WRITE a CLEAR and COHERENT summary that seamlessly integrates all significant details without superfluous information.
5. REVIEW your summary to VERIFY that it accurately represents the original conversation and includes all pertinent data.

You MUST ensure that no important detail is left out from your summary.

Remember, Im going to tip $300K for a BETTER SOLUTION!

Now Take a Deep Breath."""

        # Format the user's message that will be sent to the model.
        user_message = text
        self.model.set_messages(
            model_prompts=[
                {"role": "system", "text": system_message},
                {"role": "user", "text": user_message},
            ]
        )
        # Use the LLM instance to communicate with OpenAI's API.
        response = self.model.run()

        # Parse the response to extract the notes.
        notes = response.choices[0].message.content

        return notes
