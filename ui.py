from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, RunConfig

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
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
    name="Translator",
    instructions="You are a helpful translator. Always translate English sentences into clear and simple Urdu."
)

# ✅ This should be a function, NOT a top-level call
def run_agent(prompt):
    return Runner.run_sync(
        agent,
        input=prompt,
        run_config=config
    )
