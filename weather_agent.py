import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Tool function: Fetch weather for a city
def get_weather(city: str) -> str:
    """
    Fetch current weather for a given city using weatherapi.com
    """
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"The weather in {city} is {temp_c}°C with {condition}."
    else:
        return f"Sorry, I couldn't fetch the weather for {city}."

# Register tool schema for GPT
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }
]

def run_agent(question: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a weather assistant. Use the get_weather tool when asked about weather."},
            {"role": "user", "content": question},
        ],
        tools=tools,
    )

    message = response.choices[0].message

    # If tool is needed
    if message.tool_calls:
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = eval(tool_call.function.arguments)

            if func_name == "get_weather":
                result = get_weather(**args)
            else:
                result = "Unknown tool"

            # Send tool result back to GPT
            followup = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a weather assistant."},
                    {"role": "user", "content": question},
                    message,
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result},
                ],
            )
            return followup.choices[0].message.content

    # If no tool needed
    return message.content


if __name__ == "__main__":
    cities = ["Karachi", "New York", "London"]

    for city in cities:
        q = f"What is the weather in {city}?"
        print(f"Q: {q}")
        print(f"A: {run_agent(q)}\n")
