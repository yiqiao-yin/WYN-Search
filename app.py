from typing import Dict

from duckduckgo_search import DDGS
import google.generativeai as palm
import streamlit as st
import pandas as pd

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


# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Search 🧐</h1>
    """,
    unsafe_allow_html=True,
)


# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
domain = st.sidebar.selectbox(
    "Choose which domain you want to search:",
    ("Text", "Video", "More to come...")
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
        response = call_palm(f"{processed_user_question}")
    elif domain == "Video":
        response = video_search(prompt)
    else:
        search_results = internet_search(prompt)
        context = search_results["context"]
        urls = search_results["urls"]
        processed_user_question = f"""
            You are a search engine and you have information from the internet here: {context}.
            In addition, you have a list of URls as reference: {urls}.
            Answer the following question: {prompt} based on the information above. 
            Make sure to return URls as list of citations. 
        """
        response = call_palm(f"{processed_user_question}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
