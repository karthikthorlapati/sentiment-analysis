
import streamlit as st
import requests
import os

# Configuration
API_PORT = os.getenv("API_PORT", "8000")
# When running in Docker, 'api' is the hostname. Locally it's localhost.
# We usually use localhost for the browser to access, but Streamlit is running in a container talking to API container?
# Actually Streamlit logic runs on the server (container), so it needs to talk to the API container.
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

st.title("🐦 Twitter Sentiment Analysis")
st.markdown("Enter a tweet or text below to analyze its sentiment.")

with st.form("sentiment_form"):
    text_input = st.text_area("Text to analyze", height=150)
    submit_button = st.form_submit_button("Analyze Sentiment")

if submit_button:
    if text_input.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(f"{API_URL}/predict", json={"text": text_input})
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment.upper())
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                    if sentiment == "positive":
                        st.balloons()
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    if response.status_code == 503:
                         st.warning("The model service is unavailable. Has it been trained?")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API service. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")
