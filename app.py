import streamlit as st
from newspaper import Article
from transformers import pipeline
from textblob import TextBlob

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI News Intel", page_icon="📰", layout="centered")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_summarizer():
    # Loading the BART model for Abstractive Summarization
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# --- UI DESIGN ---
st.title("📰 Smart News Summarizer")
st.markdown("### Turn long articles into 3-bullet insights using AI.")
st.write("Enter a news URL below to analyze the content and sentiment.")

url = st.text_input("News URL:", placeholder="e.g., https://www.bbc.com/news/tech...")

# --- MAIN LOGIC ---
if st.button("Generate Report"):
    if url:
        try:
            with st.spinner('Analyzing article...'):
                # 1. Scraping
                article = Article(url)
                article.download()
                article.parse()
                
                # 2. AI Summarization (Abstractive)
                # Limits input to first 3000 chars to prevent token overflow
                text_to_summarize = article.text[:3000]
                summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
                summary_result = summary[0]['summary_text']
                
                # 3. Sentiment Analysis
                analysis = TextBlob(article.text)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.1:
                    label, color = "Positive 😊", "green"
                elif polarity < -0.1:
                    label, color = "Negative 😡", "red"
                else:
                    label, color = "Neutral 😐", "gray"

                # 4. Display Results
                st.divider()
                st.subheader(f"Title: {article.title}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Key Highlights")
                    # Cleanly format bullets
                    bullets = summary_result.split(". ")
                    for b in bullets[:3]:
                        if len(b) > 5:
                            st.write(f"• {b.strip('.')}.")

                with col2:
                    st.markdown("#### Tone Analysis")
                    st.write(f"Status: **:{color}[{label}]**")
                    st.progress((polarity + 1) / 2)
                    st.caption(f"Score: {round(polarity, 2)}")

                # 5. Export Feature
                st.divider()
                full_report = f"TITLE: {article.title}\nURL: {url}\n\nSUMMARY:\n{summary_result}\n\nSENTIMENT: {label}"
                st.download_button(
                    label="📥 Download Report",
                    data=full_report,
                    file_name="news_analysis.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"Oops! Something went wrong. Make sure the URL is valid. Error: {e}")
    else:
        st.warning("Please paste a URL first!")

# Sidebar Info
st.sidebar.title("About Project")
st.sidebar.info(
    "This app uses **NLP (Natural Language Processing)** to automate news reading. "
    "It leverages the **BART Model** for summarization and **TextBlob** for sentiment."
)