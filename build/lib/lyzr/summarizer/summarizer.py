import os
from typing import Optional, Literal
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError


class Summarizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_type: Optional[Literal["openai"]] = None,
        model: Optional[LLM] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key is None:
            raise MissingValueError("API key")
        self.model = model or get_model(
            api_key=self.api_key,
            model_type=model_type or os.environ.get("MODEL_TYPE") or "openai",
        )  # change get_model in lyzr.base.llms to accept **kwargs

    def summarize(
            self, 
            text:str,
            instructions: Optional[str] = "Summary" #Could be summary or notes or even a tweet
            ) -> str:
        '''
    Generates a concise summary or notes from the provided text using a preconfigured Large Language Model (LLM), specifically targeting OpenAI's GPT-4 model. This function is designed to streamline extensive paragraphs or conversations into a more digestible format, according to the specified instructions. It internally configures a detailed instruction set for the AI, ensuring the output captures all critical information while omitting superfluous details.

    Parameters:
    
    - `text` (str): The substantial text or conversation input that needs to be summarized or condensed.
      
    - `instructions` (Optional[str], default = "Summary"): Specifies the type of output desired. Options include "Summary" for a straightforward summarization, "Notes" for a bullet-point or outlined form, or custom instructions such as a "Tweet" for extremely concise content. This parameter influences the instruction set given to the AI, tailoring its approach to content generation.
    
    Return:
    
    - A string containing the generated summary or notes that effectively condense the essence of the original text into the desired format. This output aims to retain all pertinent information and key themes while presenting them in a clear, coherent, and logically structured manner.
    
    Example Usage:
    
    ```python
    from lyzr import Summarizer
    summarizer = Summarizer(api_key="your_openai_api_key")
    
    text = "Big paragraphs or conversations that you wish to streamline or shorten"
    summary = summarizer.summarize(text)
    print(summary)
    
    # Or for a different format:
    notes = summarizer.summarize(text, instructions='Notes')
    print(notes)
    ```
    
    This functionality leverages advanced language model capabilities for creating succinct and accurate representations of larger bodies of text, adjustable to various output instructions for enhanced utility in information processing and content creation scenarios.
        '''
        if self.model.model_name != "gpt-4":
            if self.model.model_type == "openai":
                self.model = get_model(
                    api_key=self.api_key,
                    model_type=self.model.model_type,
                    model_name="gpt-4",
                )
            else:
                raise ValueError(
                    "This function only works with the OpenAI's 'gpt-4' model."
                )

        # The system message acts as the prompt for the AI.
        system_message = f'''You are an Expert SUMMARIZER with a keen ability to CAPTURE ESSENTIAL DETAILS from extensive conversations. Your task is to CREATE a CONCISE SUMMARY of the given content, ensuring that ALL CRITICAL INFORMATION is included.

The Format of the summary should be based on these instructions: {instructions}        

Here's your step-by-step guide:

1. CAREFULLY READ through the entire conversation to fully understand the context and main points.
2. IDENTIFY and HIGHLIGHT the KEY THEMES, decisions, questions, and any action items discussed in the conversation.
3. ORGANIZE these points into a LOGICAL STRUCTURE that reflects the progression of the conversation.
4. WRITE a CLEAR and COHERENT summary that seamlessly integrates all significant details without superfluous information.
5. REVIEW your summary to VERIFY that it accurately represents the original conversation and includes all pertinent data.
6. DON'T display anything that is not relevant to the summary such as comments or instructions.

Now Take a Deep Breath.'''

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