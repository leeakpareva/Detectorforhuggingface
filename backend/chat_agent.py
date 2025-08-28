"""
AI Chat Agent with conversation memory and text-to-speech capabilities
"""
import os
from openai import OpenAI # type: ignore
import tempfile
from datetime import datetime
import json

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
client = OpenAI(api_key=api_key)

class ChatAgent:
    def __init__(self):
        """Initialize the chat agent with conversation memory"""
        self.conversation_history = []
        self.system_prompt = """You are NAVADA Assistant, an intelligent AI companion for computer vision analysis. 
        You help users understand what's in their images, answer questions about detected objects, 
        and provide insights about visual content. You're friendly, helpful, and knowledgeable about 
        computer vision, image analysis, and can discuss colors, positions, sizes, and relationships 
        between objects in images. You have access to detailed detection results including object colors, 
        positions, sizes, and confidence scores."""
        
        # Add system message to history
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Store context about current image analysis
        self.current_image_context = None
        
    def update_image_context(self, detected_objects, detailed_attributes=None):
        """Update the agent's knowledge about the current image"""
        context = f"Current image analysis shows: {', '.join(detected_objects) if detected_objects else 'no objects detected'}."
        
        if detailed_attributes:
            context += "\n\nDetailed analysis:"
            for attr in detailed_attributes:
                colors = " and ".join(attr.get('colors', ['unknown'])[:2])
                context += f"\n- {attr['label']}: {colors} color(s), {attr.get('size', 'unknown')} size, located at {attr.get('position', 'unknown')} (confidence: {attr.get('confidence', 'unknown')})"
        
        self.current_image_context = context
        
        # Add context to conversation as a system message
        self.conversation_history.append({
            "role": "system",
            "content": f"Image context update: {context}"
        })
    
    def chat(self, user_message, include_voice=True):
        """
        Process user message and return response with optional voice
        
        Args:
            user_message: The user's input message
            include_voice: Whether to generate voice response
            
        Returns:
            tuple: (text_response, voice_file_path or None)
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Keep conversation history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            # Keep system prompt and current context, remove old messages
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            recent_messages = self.conversation_history[-15:]
            self.conversation_history = system_messages + recent_messages
        
        try:
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=500
            )
            
            text_response = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": text_response
            })
            
            # Generate voice if requested
            voice_file = None
            if include_voice:
                voice_file = self.generate_voice(text_response)
            
            return text_response, voice_file
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            return error_msg, None
    
    def generate_voice(self, text):
        """Generate voice narration for text using OpenAI TTS"""
        try:
            # Generate speech using OpenAI TTS
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="mp3"
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(response.content)
                return temp_audio.name
                
        except Exception as e:
            print(f"Voice generation error: {e}")
            return None
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        messages = [msg for msg in self.conversation_history if msg["role"] in ["user", "assistant"]]
        return messages
    
    def reset_conversation(self):
        """Reset conversation history while keeping system prompt"""
        self.conversation_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        self.current_image_context = None
    
    def save_conversation(self, filepath=None):
        """Save conversation history to file"""
        if filepath is None:
            filepath = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'conversation': self.conversation_history,
                'image_context': self.current_image_context
            }, f, indent=2)
        
        return filepath
    
    def load_conversation(self, filepath):
        """Load conversation history from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.conversation_history = data['conversation']
            self.current_image_context = data.get('image_context')

# Create a global chat agent instance
chat_agent = ChatAgent()

# Helper functions for easy integration
def chat_with_agent(message, detected_objects=None, detailed_attributes=None, include_voice=True):
    """
    Simple interface to chat with the agent
    
    Args:
        message: User's message
        detected_objects: List of detected objects (optional)
        detailed_attributes: Detailed attributes from enhanced detection (optional)
        include_voice: Whether to generate voice response
        
    Returns:
        tuple: (text_response, voice_file_path or None)
    """
    # Update context if new detection results provided
    if detected_objects is not None:
        chat_agent.update_image_context(detected_objects, detailed_attributes)
    
    return chat_agent.chat(message, include_voice)

def reset_chat():
    """Reset the chat conversation"""
    chat_agent.reset_conversation()

def get_chat_history():
    """Get the current chat history"""
    return chat_agent.get_conversation_summary()