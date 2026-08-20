import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


def get_api_key():
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("Missing GOOGLE_API_KEY.")
        st.stop()
    return key


WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
    3: "Overcast", 45: "Foggy", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Freezing light drizzle", 57: "Freezing dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Freezing light rain", 67: "Freezing heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight showers", 81: "Moderate showers",
    82: "Violent showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this for factual questions, news, current events, or anything that needs up-to-date information. Do NOT use this for weather questions."""
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        resp.raise_for_status()
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.capture = False
                self.current = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self.capture = True
                    self.current = ""

            def handle_endtag(self, tag):
                if self.capture and tag == "a":
                    self.capture = False
                    if self.current.strip():
                        self.results.append(self.current.strip())

            def handle_data(self, data):
                if self.capture:
                    self.current += data

        parser = DDGParser()
        parser.feed(resp.text)
        if not parser.results:
            return f"No search results found for '{query}'."
        return "\n\n".join(parser.results[:5])
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def get_datetime(dummy: str = "") -> str:
    """Returns the current real-time date, time and day. Use this when user asks about current time, date, day, month or year. Do not pass any argument."""
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return (
            f"Current date and time in India:\n"
            f"Date: {now.strftime('%A, %d %B %Y')}\n"
            f"Time: {now.strftime('%I:%M %p')}\n"
            f"Timezone: Asia/Kolkata (IST)"
        )
    except Exception as e:
        return f"Error getting date/time: {str(e)}"


@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city. Use this when user asks about weather, temperature, humidity, wind speed, or any climate-related question. Always use this tool for weather questions, never use web search for weather."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=5&language=en"
        geo_res = requests.get(geo_url, timeout=10).json()
        if not geo_res.get("results"):
            return f"Sorry, I couldn't find the city '{city}'. Please check the spelling."
        res = geo_res["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        name = res["name"]
        country = res.get("country", "")
        admin = res.get("admin1", "")
        location_str = ", ".join(filter(None, [name, admin, country]))
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,apparent_temperature,"
            f"relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
            f"wind_gusts_10m,precipitation,weather_code,"
            f"cloud_cover,pressure_msl"
            f"&timezone=auto"
        )
        weather_res = requests.get(weather_url, timeout=10).json()
        if "current" not in weather_res:
            return f"Weather data is temporarily unavailable for {location_str}."
        curr = weather_res["current"]
        code = curr.get("weather_code", -1)
        condition = WMO_CODES.get(code, f"Unknown condition (code {code})")
        return (
            f"Current weather in {location_str}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {curr['temperature_2m']}C "
            f"(Feels like: {curr['apparent_temperature']}C)\n"
            f"Humidity: {curr['relative_humidity_2m']}%\n"
            f"Cloud Cover: {curr['cloud_cover']}%\n"
            f"Wind Speed: {curr['wind_speed_10m']} km/h "
            f"(Gusts: {curr['wind_gusts_10m']} km/h)\n"
            f"Precipitation: {curr['precipitation']} mm\n"
            f"Pressure: {curr['pressure_msl']} hPa\n"
            f"Coordinates used: {lat}N, {lon}E"
        )
    except requests.exceptions.Timeout:
        return "The weather service is taking too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return "Could not connect to the weather service. Please check your internet."
    except Exception as e:
        return f"Error getting weather: {str(e)}"


@tool
def search_documents(query: str) -> str:
    """Search through uploaded documents to find relevant information. Use this tool when the user asks questions about their uploaded files, documents, notes, or any content they have provided. Do NOT use this if no documents have been uploaded."""
    try:
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
            return "No documents have been uploaded yet. Please ask the user to upload a document first."
        docs = st.session_state.vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in the uploaded documents."
        results = "\n\n---\n\n".join(
            [f"[From: {doc.metadata.get('source', 'uploaded document')}]\n{doc.page_content}" for doc in docs]
        )
        return f"Found in uploaded documents:\n\n{results}"
    except Exception as e:
        return f"Error searching documents: {str(e)}"


def process_uploaded_file(uploaded_file, api_key):
    """Process an uploaded file: read, chunk, embed, and store in FAISS."""
    file_name = uploaded_file.name

    if file_name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = uploaded_file.read().decode("utf-8")

    if not text.strip():
        return None, "The file appears to be empty or unreadable."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": file_name}],
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, f"Processed '{file_name}': {len(chunks)} chunks created."


def create_gemini_agent():
    api_key = get_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )
    memory = MemorySaver()
    system_message = (
        "You are a sharp-witted and helpful AI assistant named 'Assistant D'. "
        "Be concise, friendly, and occasionally crack a joke. "
        "\n\nSTRICTLY follow these rules:\n"
        "\n## Tool Rules:\n"
        "1. Weather questions - ALWAYS use get_weather_data tool. NEVER use web search for weather.\n"
        "2. Time, date, day questions - ALWAYS use get_datetime tool. Call it with no arguments.\n"
        "3. Questions about uploaded documents or files - ALWAYS use search_documents tool.\n"
        "4. Factual questions, news, current events - ALWAYS use web search first.\n"
        "5. General conversation (greetings, opinions, jokes) - respond directly.\n"
        "\n## Anti-Hallucination Rules:\n"
        "6. NEVER make up facts, statistics, dates, names, or URLs.\n"
        "7. If a tool returns an error or no results, say 'I couldn't find that information' "
        "- do NOT guess or fabricate an answer.\n"
        "8. If you are unsure about something, clearly say 'I'm not 100% sure about this'.\n"
        "9. When presenting search results, stick to what the search returned. "
        "Do NOT add extra details that weren't in the results.\n"
        "10. When reporting weather, present ALL data from the tool exactly as returned. "
        "Do NOT invent additional weather details.\n"
        "11. NEVER generate fake URLs or links. Only share URLs from search results.\n"
        "12. When answering from documents, quote or closely paraphrase the source text. "
        "Do NOT add information that is not in the retrieved document chunks.\n"
    )
    agent_executor = create_react_agent(
        model=llm,
        tools=[web_search, get_weather_data, get_datetime, search_documents],
        checkpointer=memory,
        prompt=system_message,
    )
    return agent_executor
