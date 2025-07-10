import streamlit as st
from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is not set in .env")
    st.stop()

# Set up external client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Set up agent
agent = Agent(
    name="Translator",
    instructions="You are a helpful translator. Always translate English sentences into clear and simple Urdu."
)

# Define runner
def run_agent(prompt):
    return Runner.run_sync(
        agent,
        input=prompt,
        run_config=config
    )

# === Streamlit UI ===
st.title("🌐 English to Urdu Translator")

user_input = st.text_input("Enter an English sentence:")

if st.button("Translate"):
    if user_input.strip():
        try:
            result = run_agent(user_input)
            st.success("Translation:")
            st.write(result.output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text.")
