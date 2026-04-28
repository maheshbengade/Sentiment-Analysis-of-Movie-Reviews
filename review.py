import streamlit as st
import pickle

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis of Movie Reviews",
    page_icon="🎬",
    layout="centered"
)

# ------------------------------------
# Load Model
# ------------------------------------
model = pickle.load(open("movie review analysis.pkl", "rb"))

# ------------------------------------
# Load Vectorizer
# ------------------------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------------------------
# Custom CSS
# ------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1 {
    text-align: center;
    color: white;
}

textarea {
    border-radius: 10px !important;
}

.stButton button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    background-color: #4f46e5;
    color: white;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Title
# ------------------------------------
st.title("🎬 Sentiment Analysis of Movie Reviews")

st.write("Analyze whether a movie review is Positive or Negative.")

# ------------------------------------
# User Input
# ------------------------------------
review = st.text_area(
    "Enter Movie Review",
    height=200,
    placeholder="Type your review here..."
)

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("⚠ Please enter a review.")

    else:

        # Convert review to vector
        review_vector = vectorizer.transform([review])

        # Predict
        prediction = model.predict(review_vector)

        # Output
        if prediction[0] == 1:
            st.success("✅ Positive Review")
            st.balloons()

        else:
            st.error("❌ Negative Review")

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("""
<style>
.footer {
    text-align: center;
    color: #cbd5e1;
    padding: 20px;
    font-size: 16px;
}
</style>
<div class="footer">
🎬 🍿 🤖 Built with ⭐ using NLP • Machine Learning • Streamlit 🚀
</div>
""", unsafe_allow_html=True)
st.caption("")