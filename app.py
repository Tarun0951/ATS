import streamlit as st
import docx2txt
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer()


def main():
    st.title("Application Tracking System")
    
    changes = '''
    <style>
    [data-testid="stAppViewContainer"]
    {
    background-image:url(https://images.pexels.com/photos/7130872/pexels-photo-7130872.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2);
    background-size:fit;
    }
    .st-bx {
    background-color: rgba(255, 255, 255, 0.05);
    }

    /* .css-1hynsf2 .esravye2 */

    html {
    background: transparent;
    }
    div.esravye2 > iframe {
        background-color: transparent;
    }
    </style>
    '''

    # Pushing changes to the UI
    st.markdown(changes, unsafe_allow_html=True)
    
    job_desc_file = st.file_uploader("Upload job description (.docx or .pdf)", type=["docx", "pdf"])

    resume_file = st.file_uploader("Upload applicant resume (.docx or .pdf)", type=["docx", "pdf"])

    if job_desc_file and resume_file:
        job_desc_text = extract_text(job_desc_file)
        resume_text = extract_text(resume_file)

        if not job_desc_text or not resume_text:
            st.write("Text extraction failed. Please upload valid files.")
            return

        texts = [job_desc_text, resume_text]

        matrix = cv.fit_transform(texts)

        similarity_matrix = cosine_similarity(matrix)
        similarity_score = similarity_matrix[0][1]

        similarity_percentage = round(similarity_score * 100, 2)
        st.header(f"Similarity Score: {similarity_percentage}%")

        if similarity_percentage > 60:
            st.header("Resume Selected")
        else:
            st.header("Resume Not Selected")


def extract_text(file):
    text = ""
    file_extension = file.name.split(".")[-1].lower()

    if file_extension == "docx":
        text = docx2txt.process(file)
    elif file_extension == "pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        st.write(f"Unsupported file format: {file_extension}")

    return text


if __name__ == "__main__":
    main()

