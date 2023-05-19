
# Voice Assistant Application

This application allows you to interact with a voice assistant using your voice, ask questions about a PDF document, and generate images using OpenAI's DALL·E model. 

## Features

1. **Voice Assistant**: This feature allows you to interact with a voice assistant using your voice. You can specify the duration of the recording, choose the model for the assistant, and start the conversation. The assistant will respond to your queries and the conversation history can be viewed and cleared.

2. **Ask your PDF**: This feature allows you to ask questions about a PDF document. You can upload a PDF document and ask a question about it. The application will search for similar documents in the knowledge base and generate a response to your question.

3. **DALL·E Generator**: This feature allows you to generate images using OpenAI's DALL·E model. You can enter a prompt and specify the number of images to generate.

## Installation

1. Clone this repository to your local machine.
2. Install the required packages using pip:
    ```
    pip install -r requirements.txt
    ```
3. Run the application using Streamlit:
    ```
    streamlit run main.py
    ```

## Usage

1. **Voice Assistant**: Click on the 'Start Recording' button to start recording your voice. The assistant will respond to your queries and the conversation history can be viewed and cleared.

2. **Ask your PDF**: Upload a PDF document and enter a question in the text input field. Click on the 'Ask' button to get a response to your question.

3. **DALL·E Generator**: Enter a prompt in the text input field and specify the number of images to generate. Click on the 'Generate Images' button to start generating images.

## Settings

You can customize the settings of the application from the sidebar. You can choose the model for the assistant, specify the recording duration, and choose the tools to load. Advanced settings such as temperature can also be adjusted.

Absolutely, here's the updated attributions section:

## Attributions

This application uses several external libraries and resources:

- **Langchain**: The core functionalities of this application, such as the voice assistant and the PDF question-answering feature, are powered by Langchain. Langchain is a powerful tool for building conversational AI applications. More information can be found on their [website](https://www.langchain.com/).

- **OpenAI**: This application uses OpenAI's GPT-3 and GPT-4 models for natural language processing tasks, DALL·E for image generation, and Whisper for speech transcription. More information can be found on their [website](https://www.openai.com/).

- **Eleven Labs**: The voice synthesis in this application is powered by Eleven Labs. More information can be found on their [website](https://beta.elevenlabs.io/).

- **Streamlit**: This application is built using Streamlit, a powerful tool for building interactive web applications with Python. More information can be found on their [website](https://www.streamlit.com/).

- **Ask Your PDF**: The "Ask Your PDF" feature of this application is inspired by a tutorial by Alejandro Ao on YouTube. The original tutorial can be found [here](https://www.youtube.com/watch?v=wUAUdEw5oxM).