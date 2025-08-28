import os
from openai import OpenAI # type: ignore
import tempfile

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_detection(objects_list):
    """
    Sends detected objects to OpenAI and returns an explanation.
    """
    if not objects_list:
        return "No objects detected."

    prompt = f"Explain these detected objects in simple terms: {objects_list}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # new lightweight chat model
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def generate_voice(text):
    """
    Generates voice narration using OpenAI TTS with the specified voice prompt.
    Uses the prompt ID: pmpt_68af917dc0c08190b34e64407c8ec6a004645f5e56fbe8f6
    """
    try:
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # You can change this to: alloy, echo, fable, onyx, nova, or shimmer
            input=text,
            response_format="mp3"
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            return temp_audio.name
            
    except Exception as e:
        print(f"Voice generation error: {e}")
        return None