import streamlit as st
import pandas as pd
import PyPDF2
from textblob import TextBlob
import plotly.express as px
import re

st.set_page_config(page_title="Product Review Sentiment Analysis", layout="wide")

st.title("📊 Product Review Sentiment Analysis")

uploaded_file = st.file_uploader("Upload PDF with Product Reviews", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.polarity
    subjectivity = analysis.subjectivity  # 0 = objective, 1 = subjective

    if subjectivity < 0.2:           # too objective to classify
        return "Neutral", polarity
    elif polarity > 0.05:
        return "Positive", polarity
    elif polarity < -0.05:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def split_reviews(text):
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Try numbered or labeled patterns
    reviews = re.split(r'\n(?=\d+[\.\)]\s|\bReview\s*\d*\s*:|\bCustomer\b|\bUser\b)', text)

    # Fallback: double newline
    if len(reviews) <= 1:
        reviews = re.split(r'\n\s*\n', text)

    # Last resort: single newline
    if len(reviews) <= 1:
        reviews = re.split(r'\n', text)

    reviews = [r.strip() for r in reviews if r.strip() and len(r.strip()) > 30]
    return reviews

if uploaded_file is not None:
    text_content = extract_text_from_pdf(uploaded_file)

    reviews = split_reviews(text_content)

    st.write("Detected reviews:", len(reviews))

    st.success(f"✅ Extracted {len(reviews)} reviews from the PDF")

    results = []
    for i, review in enumerate(reviews):
        sentiment, polarity = get_sentiment(review)
        results.append({
            'Review_ID': i + 1,
            'Review': review[:200] + '...' if len(review) > 200 else review,
            'Full_Review': review,
            'Sentiment': sentiment,
            'Polarity_Score': round(polarity, 3)
        })

    df = pd.DataFrame(results)

    col1, col2, col3 = st.columns(3)

    positive_count = len(df[df['Sentiment'] == 'Positive'])
    negative_count = len(df[df['Sentiment'] == 'Negative'])
    neutral_count = len(df[df['Sentiment'] == 'Neutral'])

    with col1:
        st.metric("✅ Positive Reviews", positive_count, f'{(positive_count / len(df) * 100):.1f}%')

    with col2:
        st.metric("❌ Negative Reviews", negative_count, f'{(negative_count / len(df) * 100):.1f}%')

    with col3:
        st.metric("⚪ Neutral Reviews", neutral_count, f'{(neutral_count / len(df) * 100):.1f}%')

    st.divider()

    col1, col2 = st.columns(2)
    sentiment_counts = df['Sentiment'].value_counts()

    with col1:
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#00CC96',
                'Negative': '#EF553B',
                'Neutral': '#636EFA'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Count",
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#00CC96',
                'Negative': '#EF553B',
                'Neutral': '#636EFA'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        avg_polarity = df['Polarity_Score'].mean()
        st.metric("Average Polarity Score", f"{avg_polarity:.3f}")

    with col2:
        most_common = df['Sentiment'].mode()[0]
        st.metric("Most Common Sentiment", most_common)

    st.divider()

    fig_hist = px.histogram(
        df,
        x='Polarity_Score',
        nbins=10,
        title="Polarity Score Distribution",
        labels={'Polarity_Score': 'Polarity Score'},
        color_discrete_sequence=['blue']
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    st.subheader("📝 Review Details")

    filter_sentiment = st.multiselect(
        "Filter by Sentiment",
        options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )

    filtered_df = df[df['Sentiment'].isin(filter_sentiment)]

    st.dataframe(
        filtered_df[['Review_ID', 'Review', 'Sentiment', 'Polarity_Score']],
        use_container_width=True,
        height=400
    )

    st.divider()

    st.subheader("🔍 Individual Review Analysis")

    review_id = st.selectbox(
        "Select Review ID to View Full Text",
        options=df['Review_ID'].tolist()
    )

    selected_review = df[df['Review_ID'] == review_id].iloc[0]

    st.write(f"**Sentiment:** {selected_review['Sentiment']}")
    st.write(f"**Polarity Score:** {selected_review['Polarity_Score']:.3f}")
    st.write("**Full Review:**")
    st.write(selected_review['Full_Review'])

    st.divider()

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis as CSV",
        data=csv,
        file_name='sentiment_analysis_results.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Please upload a PDF file containing product reviews to begin analysis.")

    st.markdown("""
    ### How to use:
    1. Upload a PDF file containing product reviews.
    2. The app will automatically extract and analyze the reviews.
    3. View sentiment distribution and detailed statistics.
    4. Download results as a CSV

    ### Features:
    - Sentiment classification (Positive, Negative, Neutral)
    - Polarity score calculation
    - Interactive visualizations
    - Detailed review breakdown
    - Export functionality
    """)