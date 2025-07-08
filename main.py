from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, RunConfig


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key= GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


agent = Agent(
    name = "Tranlator",
    instructions = "You are a helpful translator. Always translate english sentences into clear and simple urdu."
)
resopnse = Runner.run_sync(
    agent,
    input = "My name is Muhammad Raees Alam and i am a student of GI-AIWM",
    run_config = config
    )

print(resopnse)

