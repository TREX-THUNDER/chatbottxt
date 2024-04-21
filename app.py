import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io

def check_file_format(file):
    try:
        # Read a sample of the file to check for data
        sample = file.read(1024).decode('utf-8')  # Read first 1024 bytes
        df = pd.read_csv(io.StringIO(sample))
        return True
    except Exception as e:
        return False

def get_most_similar_question(new_sentence, questions, answers, vectorizer, tfidf_matrix):
    new_tfidf = vectorizer.transform([new_sentence])
    similarities = cosine_similarity(new_tfidf, tfidf_matrix)
    most_similar_index = np.argmax(similarities)
    similarity_percentage = similarities[0, most_similar_index] * 100
    return answers[most_similar_index], similarity_percentage

def AnswerTheQuestion(new_sentence, questions, answers, vectorizer, tfidf_matrix):
    most_similar_answer, similarity_percentage = get_most_similar_question(new_sentence, questions, answers, vectorizer, tfidf_matrix)
    if similarity_percentage > 70:
        response = {"answer": most_similar_answer}
    else:
        response = {"answer": "Sorry, I don't have this information."}
    return response

def app():
    st.title("Question Answering App")

    file = st.file_uploader("Upload a CSV file with questions and answers", type=["csv"])

    if file is not None:
        try:
            # Read the uploaded file's bytes
            file_bytes = file.getvalue()

            # Create a BytesIO object from the file bytes
            file_bytesio = io.BytesIO(file_bytes)

            # Check file format before reading the entire file
            if check_file_format(file_bytesio):
                st.success("File uploaded successfully!")
                file_bytesio.seek(0)  # Reset the pointer to beginning
                df = pd.read_csv(file_bytesio)
                # Assuming the first column contains questions and the second contains answers
                questions = df.iloc[:, 0].tolist()  # Select all rows, first column
                answers = df.iloc[:, 1].tolist()  # Select all rows, second column
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(questions)

                new_sentence = st.text_input("Enter your question:")
                if st.button("Get Answer"):
                    if new_sentence:
                        response = AnswerTheQuestion(new_sentence, questions, answers, vectorizer, tfidf_matrix)
                        st.write(f"Answer: {response['answer']}")
                    else:
                        st.warning("Please enter a question.")
            else:
                st.error("Error: File seems empty or corrupt. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    app()
