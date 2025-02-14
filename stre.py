from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from decouple import config
import google.generativeai as genai

# Configure Google API Key
GOOGLE_API_KEY = config('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class StreamHandler(BaseCallbackHandler):
    """Handler for streaming the AI's response"""
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Accumulate and print each token as it's generated"""
        self.text += token  # Append the token to the text buffer
        print(token, end="", flush=True)

    def on_llm_end(self, llm, **kwargs) -> None:
        """Ensure the full response is displayed once streaming is complete"""
        if self.text:
            print(f"\nAI Response:\n{self.text}\n")
        else:
            print("\nAI Response: No response generated.\n")
        self.text = ""  # Clear the buffer for the next interaction

def create_chat():
    """Create a chat model with streaming"""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        streaming=True,
        google_api_key=GOOGLE_API_KEY
    )

def chat_stream():
    """Interactive chat with streaming responses"""
    llm = create_chat()
    stream_handler = StreamHandler()
    
    print("\nWelcome to Interactive Streaming Chat!")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        print("\nAI: ", end="", flush=True)
        
        try:
            # Get streaming response
            messages = [HumanMessage(content=user_input)]
            response = llm.invoke(
                messages,
                config={"callbacks": [stream_handler]}
            )
            
            # Directly access the content of the response
            content = response.content.replace("\n", "\n* ")  # Fix newlines
            print("\nAI Response: ", content)

        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_stream()
