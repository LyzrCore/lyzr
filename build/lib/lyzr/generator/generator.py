import os
from typing import Optional, Literal
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError


class Generator:
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

    def generate(
            self, 
            text:str,
            persona: Optional[str] = "Not Specified",
            instructions: Optional[str] = "Short Story" #Could be a Poem, a Children's story or even a tweet 
            ) -> str:
        '''
    Generates content in various styles such as a children's story, a poem, or even a tweet from the provided text using OpenAI's GPT-4 model. This function is designed to expand a byte sized prompt into a more elaborate format, according to the specified instructions.
    
    Parameters:
    
    - `text` (str): The substantial text or conversation input that needs to be elaborated or expanded.
    
    - `persona` (Optional[str], default = "Not Specified"): Specifies the persona or audience for which the content is tailored. This parameter helps in customizing the tone and style of the generated content to better resonate with the intended audience.
    
    - `instructions` (Optional[str], default = "Short Story"): Specifies the type of output desired. Options include "Mini Essay" for a detailed narrative, "Poem" for poetic content, "Children's Story" for content suitable for children, or "Tweet" for extremely concise content suitable for social media platforms like Twitter. This parameter influences the instruction set given to the AI, tailoring its approach to content generation.
    
    Return:
    
    - A string containing the generated content that effectively expands the essence of the original text into the desired format. This output aims to retain all pertinent information and key themes while presenting them in a clear, coherent, and stylistically appropriate manner.
    
    Example Usage:
    
    ```python
    from lyzr import Generator
    
    # Initialize the content generator
    generator = Generator(api_key="your_openai_api_key")
    
    # Provide the text to be condensed and specify the desired style
    text = "Prompt or idea that you want to expand upon"
    story = generator.generate(text, instructions='story')
    print(story)
    
    # You can also specify the persona for which the content is tailored
    persona = "Tech Enthusiasts"
    condensed_content = generator.generate(text, persona=persona, instructions='Tweet')
    print(condensed_content)
    ```
    
    This functionality leverages advanced language model capabilities for creating concise and accurate representations of larger bodies of text, adjustable to various output styles and tailored to specific personas for enhanced utility in information processing and content creation scenarios.
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
        system_message = f'''You are an Expert CONTENT CREATOR. Your task is to DEVELOP a VARIETY of TEXT-BASED CONTENT that could range from BLOGS to TWEETS.

The target audience of the content: {persona}

The Format of the content should be based on these instructions: {instructions}  
        
Here's how you can approach this task:

1. IDENTIFY the target audience for whom you will be creating content. Understand their interests, needs, and preferences.
2. CHOOSE the type of content you wish to create first—whether it's a blog post, tweet, article, or any other form of written communication.
3. DECIDE on the topics that will RESONATE with your audience and align with your content strategy or goals.
4. DRAFT an outline or key points for each piece of content to ensure STRUCTURE and FLOW.
5. WRITE the initial draft focusing on ENGAGEMENT and VALUE delivery for your readers or followers.
6. EMPLOY a conversational tone or formal style according to the platform and type of content you are creating.
7. EDIT and REVISE your work to enhance CLARITY, GRAMMAR, and COHERENCE before publishing.
8. DON'T display anything that is not relevant to the content such as comments or instructions.

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