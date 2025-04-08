import streamlit as st
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date, web_research_instructions
from formatting import deduplicate_and_format_sources, format_sources
from states import SummaryState, SummaryStateInput, SummaryStateOutput
from tavily import TavilyClient
import os
import json
import time
from typing import Dict, Any, List

# Load environment variables from .env file
load_dotenv()

# Set page config for wide layout and title
st.set_page_config(
    page_title="Marlene's Deep Researcher",
    page_icon="🔍✨",
    layout="wide",
)

# Custom CSS for styling with light purple and yellow theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --light-purple: #e0d0ff;
        --highlight-yellow: #fffacd;
        --dark-purple: #9370db;
        --light-text: #6a5acd;
    }
    
    /* Page background */
    .stApp {
        background-color: var(--light-purple);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--dark-purple);
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: var(--dark-purple);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid var(--dark-purple);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--dark-purple);
        color: white;
    }
    
    /* Sources card */
    .source-card {
        background-color: white;
        border-left: 4px solid var(--dark-purple);
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* Thoughts card */
    .thought-card {
        background-color: var(--highlight-yellow);
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* Summary card */
    .summary-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: var(--dark-purple) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'research_started' not in st.session_state:
    st.session_state.research_started = False
if 'research_complete' not in st.session_state:
    st.session_state.research_complete = False
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'thoughts' not in st.session_state:
    st.session_state.thoughts = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'images' not in st.session_state:
    st.session_state.images = []
if 'image_urls' not in st.session_state:
    st.session_state.image_urls = set()  # Use a set to prevent duplicate URLs
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'research_loop_count' not in st.session_state:
    st.session_state.research_loop_count = 0

# Initialize UI containers first at the global level
if 'summary_container' not in st.session_state:
    st.session_state.summary_container = None
if 'image_container' not in st.session_state:
    st.session_state.image_container = None
if 'final_summary_container' not in st.session_state:
    st.session_state.final_summary_container = None
if 'query_container' not in st.session_state:
    st.session_state.query_container = None
if 'sources_container' not in st.session_state:
    st.session_state.sources_container = None
if 'thoughts_container' not in st.session_state:
    st.session_state.thoughts_container = None

# Header
st.title("🔍 Marlene's Deep Researcher")
st.markdown("Explore topics deeply with AI-powered research")

# Function to strip thinking tokens
def strip_thinking_tokens(text: str):
    """
    Remove <think> and </think> tags and their content from the text.
    """
    thoughts = ""
    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        thoughts = text[start:end]
        text = text[:start] + text[end:]
    return thoughts, text

# Initialize models
@st.cache_resource
def load_models():
    deep_seek_model = AzureAIChatCompletionsModel(
        endpoint="https://{resource_name}.ai.azure.com/models",
        credential=os.getenv("AZURE_AI_API_KEY"),
        model_name="DeepSeek-R1",
    )
    
    gpt_model = AzureAIChatCompletionsModel(
        endpoint="https://{resource_name}.services.ai.azure.com/models",
        credential=os.getenv("AZURE_AI_API_KEY"),
        model_name="gpt-4o",
    )
    
    return deep_seek_model, gpt_model

deep_seek_model, gpt_model = load_models()

# Define graph nodes
def generate_query(state: SummaryState):
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:"),
    ]
   
    response = gpt_model.invoke(messages)
    query = json.loads(response.content)
    search_query = query['query']
    
    # Update the UI with the query
    with st.session_state.query_container:
        st.markdown(f"**Search Query:** {search_query}")
        st.session_state.queries.append(search_query)
    
    return {"search_query": search_query}

def web_research(state: SummaryState):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_results = tavily_client.search(
        state.search_query, 
        max_results=1, 
        max_tokens_per_source=1000,
        include_raw_content=False,
        include_images=True,
        search_depth='advanced'
    )
    
    # Extract images - deduplicate by URL
    if 'images' in search_results and search_results['images']:
        for image in search_results['images']:
            # Only add image if it's valid and not already in our set of URLs
            if image not in st.session_state.image_urls:
                st.session_state.image_urls.add(image)  # Add to our URL set
                st.session_state.images.append(image)   # Add to images list
    
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)
    
    formatted_sources = format_sources(search_results)
    
    # Update the UI with sources - using container properly
    with st.session_state.sources_container:
        source_html = "<div class='source-card'>"
        for source in search_results.get('results', []):
            source_html += f"<strong>{source['title']}</strong><br>"
            source_html += f"<a href='{source['url']}'>{source['url']}</a><br>"
            source_html += f"<p>{source['content'][:300]}...</p>"
            source_html += "<hr>"
        source_html += "</div>"
        st.markdown(source_html, unsafe_allow_html=True)
        st.session_state.sources.append(formatted_sources)
    
    return {
        "sources_gathered": [format_sources(search_results)], 
        "research_loop_count": state.research_loop_count + 1, 
        "web_research_results": [search_str]
    }

def summarize_sources(state: SummaryState):
    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    messages = [
        SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)
    ]
    
    response = gpt_model.invoke(messages)
    running_summary = response.content
    
    # Update the summary in the UI - using container properly
    with st.session_state.summary_container:
        st.markdown(f"<div class='summary-container'>{running_summary}</div>", unsafe_allow_html=True)
        st.session_state.summary = running_summary
    
    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState):
    # Use the model to analyze the summary and decide whether to continue research or finalize it
    result = deep_seek_model.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )

    thoughts, text = strip_thinking_tokens(result.content)
    
    # Update the thoughts in the UI
    with st.session_state.thoughts_container:
        if thoughts:
            st.markdown(f"<div class='thought-card'><strong>AI's Thinking:</strong><br>{thoughts}</div>", unsafe_allow_html=True)
            st.session_state.thoughts.append(thoughts)
    
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(text)
        # Get the follow-up query
        query = reflection_content.get('follow_up_query')
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            return {"search_query": f"Tell me more about {state.research_topic}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}

def finalize_summary(state: SummaryState):
    # Display final summary and mark research as complete
    st.session_state.research_complete = True
    
    # Use an expander that's collapsed to effectively hide the content
    # Clear with an empty write
    
    # Display the images in the UI - ensuring uniqueness
    with st.session_state.image_container:
        if len(st.session_state.images) > 0:
            st.subheader("Related Images")
            
            # Create a simpler image display layout to avoid any index errors
            # if len(st.session_state.images) == 1:
            # Just one image - display directly
            try:
                st.image(st.session_state.images[0], use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load image: {str(e)}")
            # else:
            #     # Multiple images - use a column layout with a maximum of 2 columns
            #     image_count = len(st.session_state.images)
            #     cols = st.columns(min(2, image_count))
            #     for i, img_url in enumerate(st.session_state.images):
            #         try:
            #             cols[i % len(cols)].image(img_url, use_container_width=True)
            #         except Exception as e:
            #             cols[i % len(cols)].warning(f"Could not load image: {str(e)}")
    
    # Format the final summary with sources
    final_summary = f"## Summary\n{state.running_summary}\n\n### Sources:\n"
    for source in state.sources_gathered:
        final_summary += f"{source}\n"

    # Update final summary using container properly
    with st.session_state.summary_container:
        st.session_state.summary = ""
    
    # Update final summary using container properly
    with st.session_state.final_summary_container:
        st.markdown(f"<div class='summary-container'>{final_summary}</div>", unsafe_allow_html=True)
        st.session_state.summary = final_summary
    
    return {"running_summary": final_summary}

def route_research(state: SummaryState):
    if state.research_loop_count <= 3:
        st.session_state.research_loop_count = state.research_loop_count
        return "web_research"
    else:
        return "finalize_summary"

# Build the graph
def build_research_graph():
    builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput)
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_summary)
    builder.add_node("finalize_summary", finalize_summary)

    # Add edges
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "web_research")
    builder.add_edge("web_research", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")
    builder.add_conditional_edges("reflect_on_summary", route_research)
    builder.add_edge("finalize_summary", END)
    
    return builder.compile()

# Main interface layout
col1, col2 = st.columns([2, 1])

with col1:
    # Research input area
    st.subheader("What would you like to research?")
    research_topic = st.text_input("Enter your research topic:", placeholder="e.g., Latest advancements in quantum computing")
    
    # Create placeholders for dynamic content - assign to session state variables
    st.session_state.summary_container = st.container()
    st.session_state.image_container = st.container()
    st.session_state.final_summary_container = st.container()

with col2:
    # Status and progress area
    st.subheader("Research Progress")
    status_text = st.empty()
    
    # Create placeholders for dynamic content - assign to session state variables
    st.session_state.query_container = st.container()
    st.session_state.sources_container = st.container()
    with st.session_state.sources_container:
        st.subheader("Sources")
    
    st.session_state.thoughts_container = st.container()
    with st.session_state.thoughts_container:
        st.subheader("AI's Thoughts")

# Start button
start_col1, start_col2 = st.columns([1, 3])
with start_col1:
    if st.button("✨Start Research✨", disabled=st.session_state.research_started and not st.session_state.research_complete):
        if research_topic:
            # Reset state
            st.session_state.research_started = True
            st.session_state.research_complete = False
            st.session_state.sources = []
            st.session_state.thoughts = []
            st.session_state.summary = ""
            st.session_state.images = []
            st.session_state.image_urls = set()  # Reset image URL set
            st.session_state.queries = []
            st.session_state.research_loop_count = 0
            
            # Clear containers using proper container methods
            with st.session_state.summary_container:
                st.write("")  # Clear with empty write
            
            with st.session_state.image_container:
                st.write("")  # Clear with empty write
                
            with st.session_state.final_summary_container:
                st.write("")  # Clear with empty write
                
            with st.session_state.query_container:
                st.write("")  # Clear with empty write
                
            with st.session_state.sources_container:
                st.subheader("Sources")
                
            with st.session_state.thoughts_container:
                st.subheader("AI's Thoughts")
            
            # Show status
            status_text.markdown("🔍 **Research in progress...**")
            
            # Build and run the research graph
            graph = build_research_graph()
            
            # Run the graph and update UI as it progresses - simplify progress tracking
            with st.spinner("Researching..."):
                progress_bar = status_text.progress(0.0)
                
                for event in graph.stream({"research_topic": research_topic}):
                    if event and 'research_loop_count' in event:
                        # Simple linear progress based on loop count (0 to 90%)
                        loop_count = event['research_loop_count']
                        progress_value = min(0.9, loop_count / 4)
                        progress_bar.progress(progress_value)
            
            # Research complete
            progress_bar.progress(1.0)
            status_text.markdown("✅ **Research complete!**")
        else:
            st.error("Please enter a research topic.")

with start_col2:
    # Display current status
    if st.session_state.research_started and not st.session_state.research_complete:
        st.info("Research is in progress. Please wait...")
    elif st.session_state.research_complete:
        st.success("Research complete! View your summary below.")

# Reset button
if st.session_state.research_complete:
    if st.button("Start New Research"):
        # Reset all state
        st.session_state.research_started = False
        st.session_state.research_complete = False
        st.session_state.sources = []
        st.session_state.thoughts = []
        st.session_state.summary = ""
        st.session_state.images = []
        st.session_state.image_urls = set()  # Reset image URL set
        st.session_state.queries = []
        st.session_state.research_loop_count = 0
        
        # Clear all containers
        with st.session_state.summary_container:
            st.write("")  # Clear with empty write
        with st.session_state.image_container:
            st.write("")  # Clear with empty write
        with st.session_state.final_summary_container:
            st.write("")  # Clear with empty write
        with st.session_state.query_container:
            st.write("")  # Clear with empty write
        with st.session_state.sources_container:
            st.subheader("Sources")
        with st.session_state.thoughts_container:
            st.subheader("AI's Thoughts")
        status_text.empty()