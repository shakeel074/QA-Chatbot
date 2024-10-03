import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser





from dotenv import load_dotenv
load_dotenv()
# from dotenv import find_dotenv, load_dotenv
# load_dotenv(find_dotenv(), override=True)

#Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user quries."),
        ('user', "Question: {question}")
    ]
)

def generate_response(question,engine, temperature, max_token):
    # openai.api_key = api_key
    llm = ChatOpenAI(model=engine)
    output_parser = StrOutputParser()
    chain = prompt |  llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

#StreamLit
#Title for the App
st.title("Shakeel-GPT")


#Sidebar for settings
st.sidebar.title("Settings")
# api_key= st.sidebar.text_input("Please put your OpenAi API Key", type='password')

#Select OpenAI Model
engine = st.sidebar.selectbox('Select OpenAI Model', ['gpt-4o', 'gpt-4-turbo', 'gpt-4'])

#Adjust Temperature, Token Value
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)

max_token =  st.sidebar.slider('Max Token', min_value=100, max_value=300, value=150)

#main interface for user input

st.write("Please ask your Question")
user_input=st.text_input("Enter Your Prompt")

if user_input:
    response = generate_response(user_input,engine,temperature,max_token)
    st.write(response)
# elif user_input:
#     st.warning("Please enter your  OpenAi API Key in sidebar")
else:
    st.write("Please enter your question in the text box")









# import streamlit as st
# import openai
# from langchain.chat_models import ChatOpenAI  # Correct import for LangChain
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables
# load_dotenv(find_dotenv(), override=True)

# # Prompt Template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the user's queries."),
#         ("user", "Question: {question}")
#     ]
# )

# def generate_response(question, api_key, engine, temperature, max_token):
#     openai.api_key = api_key  # Set the API key for OpenAI
#     llm = ChatOpenAI(model=engine, temperature=temperature)  # Correct initialization
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser
#     answer = chain.invoke({'question': question})
#     return answer

# # Streamlit Interface
# st.title("Shakeel-GPT")

# # Sidebar for settings
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Please put your OpenAI API Key", type='password')

# # Select OpenAI Model
# engine = st.sidebar.selectbox('Select OpenAI Model', ['gpt-4', 'gpt-4-turbo'])  # Remove 'gpt-4o'

# # Adjust Temperature and Max Token
# temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
# max_token = st.sidebar.slider('Max Token', min_value=100, max_value=300, value=150)

# # Main interface for user input
# st.write("Please ask your Question")
# user_input = st.text_input("Enter Your Prompt")

# # Generate response when user submits input
# if user_input and api_key:
#     response = generate_response(user_input, api_key, engine, temperature, max_token)
#     st.write(response)
# elif user_input:
#     st.warning("Please enter your OpenAI API Key in the sidebar.")
# else:
#     st.write("Please enter your question in the text box.")
# 