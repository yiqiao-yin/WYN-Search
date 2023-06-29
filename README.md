# WYN Search 🧐

This is a simple web application called "WYN Search" that uses natural language processing and search engine APIs to provide search results and generate responses based on user queries.

## Description

WYN Search is built using Python and Streamlit, a popular framework for building web applications. It integrates two APIs: DuckDuckGo Search (DDGS) and Palm's Generative AI model. The application allows users to input their queries and receive responses generated by the AI model. Additionally, it provides search results from DuckDuckGo to assist in generating the responses.

## Installation

To run the application locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repo.git`
2. Navigate to the project directory: `cd your-repo`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Set up the necessary API keys:
   - Obtain a Palm API key from the Palm website.
   - Set the API key as an environment variable or update the `palm_api_key` variable in the `app.py` file.
   - Optionally, you can set up a Streamlit secrets file to store the API key securely.
5. Run the application: `streamlit run app.py`
6. Open your web browser and go to `http://localhost:8501` to access the application.

## Usage

Upon running the application, you will see the WYN Search user interface with a chat-like interface. You can interact with the application by following these steps:

1. Enter your query or prompt in the input field labeled "What is up?".
2. Press Enter or click the "Send" button to submit your query.
3. The application will display your query as a user message in the chat container.
4. The application will generate a response based on your query using the Palm AI model.
5. The response will be displayed in the chat container as an assistant message.
6. The application will also provide search results from DuckDuckGo based on your query. The context of the search results and a list of URLs will be included in the prompt for the AI model.
7. The assistant's response will be generated by the AI model and displayed in the chat container.
8. You can continue the conversation by entering new queries in the input field.

## Contributing

Contributions to WYN Search are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue on the repository. If you'd like to contribute code, you can fork the repository, make your changes, and submit a pull request.

When contributing, please ensure that your code adheres to the existing coding style and conventions. Include appropriate documentation and test your changes thoroughly.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for both commercial and non-commercial purposes. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The application utilizes the DuckDuckGo Search API to fetch search results. See their [official documentation](https://duckduckgo.com/api) for more information.
- The application incorporates Palm's Generative AI model to generate responses. Visit the [Palm website](https://palm-ml.com/) to learn more about their API and services.
- This project was inspired by the power of natural language processing and AI in providing intelligent search capabilities.