import streamlit as st

st.title("Fake News Detector")

user_input = st.text_area("Paste or type a news article below:")

if st.button("Check if Fake or Real"):
    st.write("You entered:")
    st.write(user_input)