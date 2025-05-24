import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import os
import groq
import concurrent.futures
from utils.pdf_processor import extract_text, chunk_text
from utils.rag_handler import save_index, load_index, retrieve_chunks
from pdf2image import convert_from_bytes
from PIL import Image

client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
st.set_page_config(page_title="PDF RAG Summarizer", layout="wide")

# Session state initialization
for key in ["pages", "pdf_name", "index_dir", "chunk_to_page_map", "pdf_data"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "pages" else None

st.title("PDF RAG Summarizer with Groq Models")
os.makedirs('data', exist_ok=True)
os.makedirs('utils', exist_ok=True)

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and st.session_state.pdf_name != uploaded_file.name:
    with st.spinner("Processing PDF..."):
        st.session_state.pdf_data = uploaded_file.getvalue()
        st.session_state.pages = extract_text(uploaded_file)
        chunks, st.session_state.chunk_to_page_map = chunk_text(st.session_state.pages)
        st.session_state.pdf_name = uploaded_file.name.replace('.pdf', '')
        st.session_state.index_dir = save_index(
            chunks,
            st.session_state.pdf_name,
            st.session_state.chunk_to_page_map
        )
        st.success(f"PDF processed successfully! {len(st.session_state.pages)} pages indexed.")

# Sidebar
with st.sidebar:
    st.header("📄 PDF Pages")
    if st.session_state.pdf_data:
        try:
            images = convert_from_bytes(st.session_state.pdf_data, dpi=150)
            total_pages = len(images)
            st.subheader("🔍 Jump to Page")
            jump_page = st.number_input(f"Enter page number (1–{total_pages})", 1, total_pages, 1, 1)
            st.image(images[jump_page - 1], caption=f"Page {jump_page}", use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔽️ All Pages")
            for i, img in enumerate(images):
                with st.expander(f"Page {i + 1}"):
                    st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not render PDF: {e}")
    else:
        st.info("Please upload a PDF to preview.")

# Main content
if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        primary_model = st.selectbox(
            "Select Primary Model",
            [
                "Llama 4 Maverick (Best Quality)",
                "Deepseek 70B (Reasoning)",
                "Llama 3 70B (Balanced)"
            ],
            index=0
        )

    with col2:
        compare_models = st.checkbox("Compare with another model")
        if compare_models:
            available_models = [
                "Llama 4 Maverick (Best Quality)",
                "Deepseek 70B (Reasoning)",
                "Llama 3 70B (Balanced)"
            ]
            available_models.remove(primary_model)
            secondary_model = st.selectbox("Select Secondary Model", available_models)
            st.warning("Comparing models may double API costs")

    query = st.text_input("Ask a question about the PDF", placeholder="Type your question here...")
    submit_button = st.button("Submit")

    if submit_button:
        if not query.strip():
            st.warning("❗ Please enter a question before submitting.")
        elif not st.session_state.pdf_name:
            st.error("❌ No PDF detected - please upload a valid file")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
        else:
            with st.spinner("🤖 Generating response..."):
                index, chunks, chunk_to_page_map = load_index(st.session_state.index_dir)
                retrieved_chunks, page_numbers = retrieve_chunks(query, index, chunks, chunk_to_page_map)

                if not retrieved_chunks:
                    st.warning("⚠️ Query not PDF-related - please ask about the uploaded document")
                else:
                    context = "\n\n".join(retrieved_chunks)

                    model_map = {
                        "Llama 4 Maverick (Best Quality)": "meta-llama/llama-4-maverick-17b-128e-instruct",
                        "Deepseek 70B (Reasoning)": "deepseek-r1-distill-llama-70b",
                        "Llama 3 70B (Balanced)": "llama3-70b-8192"
                    }

                    model_params = {
                        "meta-llama/llama-4-maverick-17b-128e-instruct": {"temperature": 0.3, "max_tokens": 4000},
                        "deepseek-r1-distill-llama-70b": {"temperature": 0.2, "max_tokens": 4000},
                        "llama3-70b-8192": {"temperature": 0.5, "max_tokens": 3000}
                    }

                    def get_groq_response(model_name):
                        prompt = f"""You are a helpful AI assistant that summarizes information from PDF documents.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based ONLY on the information in the context. If the answer cannot be found in the context, state that clearly."""
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=model_params[model_name]["temperature"],
                            max_tokens=model_params[model_name]["max_tokens"]
                        )
                        return response.choices[0].message.content

                    def parse_deepseek_response(response):
                        if "<think>" in response and "</think>" in response:
                            think_start = response.find("<think>") + len("<think>")
                            think_end = response.find("</think>")
                            reasoning = response[think_start:think_end].strip()
                            summary = response[think_end + len("</think>"):].strip()
                            return reasoning, summary
                        return None, response

                    if compare_models:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future_primary = executor.submit(get_groq_response, model_map[primary_model])
                            future_secondary = executor.submit(get_groq_response, model_map[secondary_model])
                            primary_response = future_primary.result()
                            secondary_response = future_secondary.result()

                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown(f"#### 🔹 {primary_model}")
                            reasoning, summary = parse_deepseek_response(primary_response) if "Deepseek" in primary_model else (None, primary_response)
                            if reasoning:
                                with st.expander("🤠 Deepseek Thinking"):
                                    st.markdown(reasoning)
                            st.markdown(f"""
                                <div style='padding:1em;background:#f1f3f6;border-radius:12px'>
                                {summary}<br><br><i><b>Source: Pages {', '.join(page_numbers)}</b></i>
                                </div>
                            """, unsafe_allow_html=True)

                        with col_b:
                            st.markdown(f"#### 🔹 {secondary_model}")
                            reasoning, summary = parse_deepseek_response(secondary_response) if "Deepseek" in secondary_model else (None, secondary_response)
                            if reasoning:
                                with st.expander("🤠 Deepseek Thinking"):
                                    st.markdown(reasoning)
                            st.markdown(f"""
                                <div style='padding:1em;background:#f1f3f6;border-radius:12px'>
                                {summary}<br><br><i><b>Source: Pages {', '.join(page_numbers)}</b></i>
                                </div>
                            """, unsafe_allow_html=True)

                        st.divider()
                        st.metric("Token Difference", f"{abs(len(primary_response) - len(secondary_response))} tokens")

                    else:
                        response = get_groq_response(model_map[primary_model])
                        reasoning, summary = parse_deepseek_response(response) if "Deepseek" in primary_model else (None, response)
                        if reasoning:
                            with st.expander("🤠 Deepseek Thinking"):
                                st.markdown(reasoning)
                        st.markdown(f"""
                            <div style='padding:1.2em;background:#e9f5e9;border-left:5px solid #34a853;border-radius:8px'>
                            {summary}<br><br><i><b>Source: Pages {', '.join(page_numbers)}</b></i>
                            </div>
                        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("PDF RAG Summarizer with Groq Models | v1.0")
