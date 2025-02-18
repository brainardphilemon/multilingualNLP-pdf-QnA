import streamlit as st
import pdfplumber
import re
from io import BytesIO
from google import genai
from google.genai import types
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# ---------------------------
# Initialize External APIs & Tools
# ---------------------------
translator = Translator()

# Initialize Gemini API client (replace with your actual API key)
client = genai.Client(api_key="AIzaSyB8GJ0UWeKVAdHu8mRzDGDgxWcwX7eyokI")
safety_settings = None  # Update as necessary.

# Initialize T5-based question generator.
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")


# ---------------------------
# Function Definitions
# ---------------------------
def extract_clean_text_chunks_from_pdf(pdf_file, chunk_size=1000):
    """
    Extract and clean text from the uploaded PDF file.
    The text is cleaned (removing URLs, extra whitespace) and then translated into English.
    The resulting text is broken into chunks of approximately chunk_size characters.
    """
    text_chunks = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Remove URLs and extra whitespace.
                page_text = re.sub(r'http\S+|www\S+|file:\S+|\S+\.html', '', page_text)
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                # Translate page text into English.
                page_text = translator.translate(page_text, dest='en').text
                # Break the page text into chunks.
                for i in range(0, len(page_text), chunk_size):
                    text_chunks.append(page_text[i:i + chunk_size])
    return text_chunks


def retrieve_relevant_chunks(prompt, text_chunks, top_n=5):
    """
    Find the most relevant text chunks for a given prompt using TF-IDF cosine similarity.
    """
    vectorizer = TfidfVectorizer().fit_transform([prompt] + text_chunks)
    cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_n_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [text_chunks[i] for i in top_n_indices]


def generate_questions_t5(text, num_questions=10):
    """
    Generate unique brief questions using the T5-based question generation model.
    """
    prompt_text = "generate brief question: " + text
    questions = question_generator(prompt_text, max_length=100, num_beams=5, num_return_sequences=num_questions)
    unique_questions = set(q['generated_text'].strip().replace("\n", " ") for q in questions)
    return list(unique_questions)


def generate_questions_gemini(text, num_questions=10):
    """
    Generate unique brief questions using Gemini.
    Gemini is instructed to output ONLY the questions labeled 1 through num_questions.
    """
    prompt = (
        f"Generate exactly {num_questions} brief questions about the following text. "
        f"Return only the questions, labeled with numbers 1 through {num_questions}, and nothing else. "
        f"Do not provide any introductions or summaries.\n\n"
        f"Text:\n{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    questions_list = []
    if response and response.text.strip():
        lines = response.text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(f"{len(questions_list) + 1}.")):
                cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                questions_list.append(cleaned_line)
    return questions_list


def generate_answers_for_questions_gemini(questions, context):
    """
    Aggregate multiple questions into a single prompt and call Gemini to get answers.
    Answers are expected to be labeled with their corresponding question number.
    """
    prompt = (
        "Answer the following questions concisely (4-5 lines max) based on the context provided. "
        "Provide each answer labeled with its corresponding question number.\n\n"
        f"Context: {context}\n\nQuestions:\n"
    )
    for i, q in enumerate(questions, start=1):
        prompt += f"{i}. {q}\n"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )

    answers = {}
    if response and response.text.strip():
        lines = response.text.strip().split('\n')
        current_index = None
        current_answer = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and line[1] == '.':
                if current_index is not None and current_answer:
                    answers[questions[current_index - 1]] = "\n".join(current_answer).strip()
                try:
                    parts = line.split('.', 1)
                    current_index = int(parts[0])
                    current_answer = [parts[1].strip()] if len(parts) > 1 else []
                except ValueError:
                    continue
            else:
                if current_index is not None:
                    current_answer.append(line)
        if current_index is not None and current_answer:
            answers[questions[current_index - 1]] = "\n".join(current_answer).strip()
    return answers


def generate_summary_gemini(text, word_count=50):
    """
    Generate a summary of the given text in exactly word_count words using Gemini.
    """
    prompt = (
        f"Summarize the following text in exactly {word_count} words. "
        "Do not include any additional commentary or explanations.\n\n"
        f"Text:\n{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    summary = response.text.strip() if response and response.text.strip() else ""
    return summary


def create_summary_text(text, word_count=50, output_lang="en"):
    """
    Generate a summary using Gemini, and translate it to the chosen language.
    """
    summary = generate_summary_gemini(text, word_count=word_count)
    # Translate summary if needed.
    if output_lang != "en":
        summary = translator.translate(summary, dest=output_lang).text
    return summary


def generate_followup_answer(selected_question, original_answer, followup_question, summary_text, aggregated_context):
    """
    Generate an answer to the follow-up question using Gemini.
    """
    followup_prompt = (
        "Answer the following follow-up question concisely (4-5 lines max) based on the context provided. "
        f"Original Question: {selected_question}\n"
        f"Original Answer: {original_answer}\n"
        f"Follow-up question: {followup_question}\n\n"
        f"Context (50-word PDF summary): {summary_text}\n"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[followup_prompt],
        config=types.GenerateContentConfig(safety_settings=safety_settings)
    )
    new_answer = response.text.strip() if response and response.text.strip() else "I'm sorry, I couldn't generate an answer at this time."
    return new_answer


# ---------------------------
# Main Processing Function
# ---------------------------
def process_pdf(pdf_file, prompt, use_t5, output_lang):
    # Extract text chunks (translated to English).
    text_chunks = extract_clean_text_chunks_from_pdf(pdf_file)

    # Retrieve the most relevant chunks for the prompt.
    relevant_chunks = retrieve_relevant_chunks(prompt, text_chunks, top_n=5)
    aggregated_text = " ".join(relevant_chunks)

    # Generate questions.
    if use_t5:
        all_questions = generate_questions_t5(aggregated_text, num_questions=10)
    else:
        all_questions = generate_questions_gemini(aggregated_text, num_questions=10)

    # Use the entire PDF content as context for answers.
    aggregated_context = " ".join(text_chunks)
    answers = generate_answers_for_questions_gemini(all_questions, aggregated_context)
    valid_questions = {q: answers[q] for q in all_questions if q in answers and answers[q]}

    # Generate summary text.
    summary_text = create_summary_text(aggregated_context, word_count=50, output_lang=output_lang)

    return aggregated_context, valid_questions, summary_text


# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("PDF Q&A & Follow-Up App")
st.write(
    "This app extracts text from a PDF, generates questions and answers (using either a T5 or Gemini model), provides a summary, and allows you to ask follow-up questions.")

# Sidebar Inputs
st.sidebar.header("Input Options")
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    "Arabic": "ar"
}
selected_language = st.sidebar.selectbox("Choose your language", list(language_options.keys()))
output_lang = language_options[selected_language]

pdf_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])
prompt_input = st.sidebar.text_input("Enter your prompt for question generation", "Enter your prompt here")
use_t5 = st.sidebar.radio("Use T5 model for question generation?", ("Yes", "No")) == "Yes"

if st.sidebar.button("Process PDF"):

    if not pdf_file or not prompt_input:
        st.error("Please upload a PDF and enter a prompt.")
    else:
        with st.spinner("Processing PDF..."):
            # Process the PDF and store results in session state.
            aggregated_context, valid_questions, summary_text = process_pdf(pdf_file,
                                                                            translator.translate(prompt_input,
                                                                                                 dest="en").text if output_lang != "en" else prompt_input,
                                                                            use_t5,
                                                                            output_lang)
            st.session_state.aggregated_context = aggregated_context
            st.session_state.valid_questions = valid_questions
            st.session_state.summary_text = summary_text
            st.session_state.use_t5 = use_t5

        st.success("PDF processed successfully!")

        # Display generated questions and answers.
        st.subheader("Generated Questions & Answers")
        if valid_questions:
            for idx, (question, answer) in enumerate(valid_questions.items(), start=1):
                # Translate question/answer back to the chosen language if necessary.
                q_display = translator.translate(question, dest=output_lang).text if output_lang != "en" else question
                a_display = translator.translate(answer, dest=output_lang).text if output_lang != "en" else answer
                st.markdown(f"**Q{idx}: {q_display}**")
                st.markdown(f"**A{idx}: {a_display}**")
                st.write("---")
        else:
            st.info("No relevant questions could be generated from the provided PDF context.")

        # Display PDF summary.
        st.subheader("PDF Summary (50 words)")
        st.write(summary_text)

        # Initialize follow-up session state variables.
        if "selected_question" not in st.session_state and valid_questions:
            # Preselect the first question.
            first_question = list(valid_questions.keys())[0]
            st.session_state.selected_question = first_question
            st.session_state.original_answer = valid_questions[first_question]

# ---------------------------
# Follow-Up Mode
# ---------------------------
if "valid_questions" in st.session_state and st.session_state.valid_questions:
    st.subheader("Follow-Up Q&A")
    st.write("Select a generated question to ask a follow-up, or choose to generate a new random question.")

    # List the questions for selection.
    question_options = list(st.session_state.valid_questions.keys())
    question_options_display = [translator.translate(q, dest=output_lang).text if output_lang != "en" else q
                                for q in question_options]
    # Add an option for a new random question.
    question_options_display.insert(0, "New Random Question")

    selected_option = st.selectbox("Choose a question", question_options_display)

    # If "New Random Question" is chosen, generate one.
    if selected_option == "New Random Question":
        with st.spinner("Generating a new random question..."):
            if st.session_state.use_t5:
                new_question = generate_questions_t5(st.session_state.aggregated_context, num_questions=1)[0]
            else:
                new_question = generate_questions_gemini(st.session_state.aggregated_context, num_questions=1)[0]
            new_answer_dict = generate_answers_for_questions_gemini([new_question], st.session_state.aggregated_context)
            new_answer = new_answer_dict.get(new_question, "I'm sorry, I couldn't generate an answer at this time.")
            st.session_state.selected_question = new_question
            st.session_state.original_answer = new_answer
            st.success("New random question generated!")
            st.markdown(
                f"**Question:** {translator.translate(new_question, dest=output_lang).text if output_lang != 'en' else new_question}")
            st.markdown(
                f"**Answer:** {translator.translate(new_answer, dest=output_lang).text if output_lang != 'en' else new_answer}")
    else:
        # Map the displayed text back to the original question.
        index = question_options_display.index(selected_option) - 1
        selected_question = question_options[index]
        st.session_state.selected_question = selected_question
        st.session_state.original_answer = st.session_state.valid_questions[selected_question]
        st.markdown(
            f"**Selected Question:** {translator.translate(selected_question, dest=output_lang).text if output_lang != 'en' else selected_question}")

    # Input for follow-up question.
    followup_input = st.text_input("Enter your follow-up question (or type 'new' to generate a new random question):",
                                   "")
    if st.button("Submit Follow-Up"):
        if followup_input.strip().lower() == "new":
            # Generate a new random question.
            with st.spinner("Generating a new random question..."):
                if st.session_state.use_t5:
                    new_question = generate_questions_t5(st.session_state.aggregated_context, num_questions=1)[0]
                else:
                    new_question = generate_questions_gemini(st.session_state.aggregated_context, num_questions=1)[0]
                new_answer_dict = generate_answers_for_questions_gemini([new_question],
                                                                        st.session_state.aggregated_context)
                new_answer = new_answer_dict.get(new_question, "I'm sorry, I couldn't generate an answer at this time.")
                st.session_state.selected_question = new_question
                st.session_state.original_answer = new_answer
                st.success("New random question generated!")
                st.markdown(
                    f"**Question:** {translator.translate(new_question, dest=output_lang).text if output_lang != 'en' else new_question}")
                st.markdown(
                    f"**Answer:** {translator.translate(new_answer, dest=output_lang).text if output_lang != 'en' else new_answer}")
        elif followup_input.strip():
            # Translate follow-up into English if needed.
            followup_eng = translator.translate(followup_input,
                                                dest="en").text if output_lang != "en" else followup_input
            with st.spinner("Generating follow-up answer..."):
                new_answer = generate_followup_answer(
                    st.session_state.selected_question,
                    st.session_state.original_answer,
                    followup_eng,
                    st.session_state.summary_text,
                    st.session_state.aggregated_context
                )
            st.markdown("**Follow-Up Answer:**")
            st.write(translator.translate(new_answer, dest=output_lang).text if output_lang != "en" else new_answer)
        else:
            st.warning("Please enter a follow-up question.")