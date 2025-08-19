import streamlit as st
from nltk.chat.util import Chat, reflections

pairs = [
    [r"(.*)my name is (.*)", ["Hello %2, How are you today ?"]],
    [r"(.*)help(.*)", ["I can help you "]],
    [r"(.*)your name\??", ["My name is thecleverprogrammer, but you can call me robot."]],
    [r"how are you (.*)\?", ["I'm doing very well", "I am great!"]],
    [r"sorry (.*)", ["It's alright", "It's OK, never mind that"]],
    [r"i'm (.*) (good|well|okay|ok)", ["Nice to hear that", "Alright, great!"]],
    [r"(hi|hey|hello|hola|holla)(.*)", ["Hello", "Hey there"]],
    [r"what (.*) want ?", ["Make me an offer I can't refuse"]],
    [r"(.*)created(.*)", ["prakash created me using Python's NLTK library", "top secret ;)"]],
    [r"(.*) (location|city)", ['Hyderabad, India']],
    [r"(.*)raining in (.*)", ["No rain in the past 4 days here in %2", "In %2 there is a 50% chance of rain"]],
    [r"how (.*) health (.*)", ["Health is important, but I don't need to worry, I'm a computer"]],
    [r"(.*)(sports|game|sport)(.*)", ["I'm a big fan of Cricket"]],
    [r"who (.*) (Cricketer|Batsman)?", ["Virat Kohli"]],
    [r"(.*)", ["I'm not sure I understand. Can you rephrase?"]]
]

chatbot = Chat(pairs, reflections)

st.title("🤖 NLTK Chatbot")
st.write("Type something and I'll reply!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Say something...")

if user_input:
    if user_input.lower() == "quit":
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", "Bye for now. See you soon :)"))
    else:
        response = chatbot.respond(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

# Display chat
for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
