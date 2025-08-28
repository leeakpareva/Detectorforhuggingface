import os
from openai import OpenAI  # type: ignore
import tempfile

# Lazily initialized OpenAI client to avoid import-time errors when the
# API key isn't configured. Previously this module attempted to create the
# client on import and raised a ``ValueError`` if ``OPENAI_API_KEY`` was
# missing, which prevented the rest of the application from running (and
# broke tests that don't require the API). The client is now created only
# when needed.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client instance.

    Raises:
        ValueError: If the ``OPENAI_API_KEY`` environment variable is not set.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required but not set")
        _client = OpenAI(api_key=api_key)
    return _client


def explain_detection(objects_list):
    """Send detected objects to OpenAI and return an explanation."""
    if not objects_list:
        return "No objects detected."

    prompt = f"Explain these detected objects in simple terms: {objects_list}"

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # new lightweight chat model
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def generate_voice(text):
    """Generate voice narration using OpenAI's TTS service."""
    try:
        client = _get_client()

        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # You can change this to: alloy, echo, fable, onyx, nova, or shimmer
            input=text,
            response_format="mp3",
        )

        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            return temp_audio.name

    except Exception as e:
        print(f"Voice generation error: {e}")
        return None
