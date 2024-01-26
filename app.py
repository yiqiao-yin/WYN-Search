from typing import Dict

import google.generativeai as palm
import openai
import pandas as pd
import streamlit as st
from duckduckgo_search import DDGS

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from typing import List, Dict, Any


palm_api_key = st.secrets["PALM_API_KEY"]
palm.configure(api_key=palm_api_key)


def call_palm(prompt: str) -> str:
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0,
        max_output_tokens=800,
    )
    return completion.result


openai.api_key = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI()


def call_chatgpt(query: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generates a response to a query using the specified language model.

    Args:
        query (str): The user's query that needs to be processed.
        model (str, optional): The language model to be used. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The generated response to the query.
    """

    # Prepare the conversation context with system and user messages.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}."},
    ]

    # Use the OpenAI client to generate a response based on the model and the conversation context.
    response: Any = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract the content of the response from the first choice.
    content: str = response.choices[0].message.content

    # Return the generated content.
    return content


def internet_search(prompt: str) -> Dict[str, str]:
    content_bodies = []
    list_of_urls = []
    with DDGS() as ddgs:
        i = 0
        for r in ddgs.text(prompt, region="wt-wt", safesearch="Off", timelimit="y"):
            if i <= 5:
                content_bodies.append(r["body"])
                list_of_urls.append(r["href"])
                i += 1
            else:
                break

    return {"context": content_bodies, "urls": list_of_urls}


def video_search(prompt: str) -> pd.DataFrame:
    data = []
    with DDGS() as ddgs:
        keywords = "tesla"
        ddgs_videos_gen = ddgs.videos(
            keywords,
            region="wt-wt",
            safesearch="Off",
            timelimit="w",
            resolution="high",
            duration="medium",
        )
        for r in ddgs_videos_gen:
            data.append({"content": r["content"], "description": r["description"]})

    data = pd.DataFrame(data).to_markdown()

    return data


SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]


def call_langchain(prompt: str) -> str:
    llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    tools = load_tools(
        ["serpapi", "llm-math"], llm=llm, serpapi_api_key=SERPAPI_API_KEY
    )
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    output = agent.run(prompt)

    return output


# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>Web Search 🧐</h1>
    """,
    unsafe_allow_html=True,
)


# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model = st.sidebar.selectbox(
    "Choose which language model do you want to use:",
    ("Langchain Agent", "GPT", "Palm"),
)
domain = st.sidebar.selectbox(
    "Choose which domain you want to search:", ("Text", "Video", "More to come...")
)
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Next item ... ")
clear_button = st.sidebar.button("Clear Conversation", key="clear")
st.sidebar.markdown(
    "@ [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)


# reset everything
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state["number_tokens"] = []
    st.session_state["domain_name"] = []
    counter_placeholder.write(f"Next item ...")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("Enter key words here."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response
    if domain == "Text":
        if model in ["GPT", "Palm"]:
            search_results = internet_search(prompt)
            context = search_results["context"]
            urls = search_results["urls"]
            processed_user_question = f"""
                Here is a url: {urls}
                Here is user question or keywords: {prompt}
                Here is some text extracted from the webpage by bs4:
                ---------
                {context[0:2]}
                ---------

                Web pages can have a lot of useless junk in them. 
                For example, there might be a lot of ads, or a 
                lot of navigation links, or a lot of text that 
                is not relevant to the topic of the page. We want 
                to extract only the useful information from the text.

                You can use the url and title to help you understand 
                the context of the text.
                Please extract only the useful information from the text. 
                Try not to rewrite the text, but instead extract 
                only the useful information from the text.

                Make sure to return URls as list of citations.
            """
        if model == "GPT":
            response = call_chatgpt(f"{processed_user_question}")
        elif model == "Palm":
            response = call_palm(f"{processed_user_question}")
        elif model == "Langchain Agent":
            response = call_langchain(f"{prompt}")
        else:
            response = call_chatgpt(f"{processed_user_question}")
    elif domain == "Video":
        response = video_search(prompt)
    else:
        search_results = internet_search(prompt)
        context = search_results["context"]
        urls = search_results["urls"]
        processed_user_question = f"""
            Here is a url: {urls}
            Here is user question or keywords: {prompt}
            Here is some text extracted from the webpage by bs4:
            ---------
            {context}
            ---------

            Web pages can have a lot of useless junk in them. 
            For example, there might be a lot of ads, or a 
            lot of navigation links, or a lot of text that 
            is not relevant to the topic of the page. We want 
            to extract only the useful information from the text.

            You can use the url and title to help you understand 
            the context of the text.
            Please extract only the useful information from the text. 
            Try not to rewrite the text, but instead extract 
            only the useful information from the text.

            Make sure to return URls as list of citations.
        """
        if model == "GPT":
            response = call_chatgpt(f"{processed_user_question}")
        elif model == "Palm":
            response = call_palm(f"{processed_user_question}")
        else:
            response = call_chatgpt(f"{processed_user_question}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
