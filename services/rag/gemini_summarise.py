import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pathlib
import httpx

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("TOKEN")

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Retrieve and encode the PDF byte
def summarize(filepath: str, model: str = "gemini-2.5-flash") -> str:
    """
    Summarizes a PDF document using the specified model.
    
    :param filepath: Path to the PDF file.
    :param model: The model to use for summarization.
    :return: Summary of the document.
    """
    filepath = pathlib.Path(filepath)
    pdf_bytes = filepath.read_bytes()
    
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type='application/pdf',
            ),
            "Summarize this legal document"
        ]
    )
    print(f"Response from model {model}: {response.text}")
    return response.text

#out = summarize('/home/biscuitbobby/Documents/mcp/services/docs/Independent_Sugar_Corporation_Limited_vs_Girish_Sriram_Juneja_on_29_January_2025.PDF')
#print(out)
