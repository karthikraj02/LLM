import streamlit as st

# Title
st.title("Welcome to My Streamlit App")

# Input box
user_input = st.text_input("Type something:")

# Display the input text
if user_input:
    st.write(f"You typed: {user_input}")