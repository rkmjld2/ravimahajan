#blood_report_analyzer_groq.py
#Version with Groq API - suitable for Streamlit Cloud / GitHub
import streamlit as st
import pandas as pd
from io import StringIO
import time
from datetime import datetime
#────────────────────────────────────────────────
#LangChain & embeddings (cloud-friendly)
#────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq
#Groq client
from groq import Groq

st.set_page_config(page_title="Blood Report Analyzer • Groq", layout="wide")

#────────────────────────────────────────────────
#Groq API Key handling
#────────────────────────────────────────────────
# Sidebar - Groq API Key (always show input on cloud too)
with st.sidebar:
    st.markdown("### Groq API Key")

    api_key = None

    # 1. Try to load from Streamlit secrets (best for cloud deployment)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("Using API key from secrets ✓")
    except:
        pass

    # 2. If no secret found → show input field (works both local and cloud)
    if not api_key:
        api_key_input = st.text_input(
            "Enter Groq API key",
            type="password",
            placeholder="gsk_...",
            value="",
            key="groq_api_input"
        )
        if api_key_input:
            api_key = api_key_input
            st.success("Key entered manually ✓")
        else:
            st.warning("Please enter your Groq API key above to use the app.")
            st.stop()

    # Safety check
    if not api_key:
        st.error("No valid Groq API key found.")
        st.stop()

    # Store in session state so the rest of the app can use it
    st.session_state.groq_api_key = api_key

    st.markdown("---")
    st.caption("Model: llama-3.3-70b-versatile via Groq")

#────────────────────────────────────────────────
#Embeddings (works everywhere - no local Ollama needed)
#────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

#────────────────────────────────────────────────
#Session state
#────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

#────────────────────────────────────────────────
#UI
#────────────────────────────────────────────────
st.title("🩸 Blood Report Analyzer – Groq Edition")
st.caption("Paste → Edit table → Process → Ask questions • Powered by Groq • Internet required")
tab1, tab2 = st.tabs(["📊 Paste & Edit Table", "ℹ️ How to use"])

with tab1:
    st.markdown(
        "Paste your blood report table (from PDF, lab website, WhatsApp, Excel, etc.)\\n"
        "Best results when columns are separated by **comma**, **tab** or **spaces**."
    )

    raw_text = st.text_area(
        "1. Paste your report table here",
        height=240,
        value="""Test,Result,Unit,Reference Range,Flag
Hemoglobin,12.4,g/dL,13.0 - 17.0,L
WBC,8.2,10^3/µL,4.0 - 11.0,
Glucose (Fasting),102,mg/dL,70 - 99,H
Creatinine,1.1,mg/dL,0.6 - 1.2,
ALT,45,U/L,7 - 56,
Total Cholesterol,210,mg/dL,<200,H""",
        help="Copy table from PDF viewer, lab portal, Excel or text message"
    )

    if st.button("2. Parse text → Show editable table", type="primary", use_container_width=True):
        if raw_text.strip():
            try:
                df = pd.read_csv(StringIO(raw_text), sep=None, engine="python", on_bad_lines="skip")
                df = df.dropna(how="all")
                st.session_state.df = df
                st.success(f"Parsed successfully — {len(df)} rows found")
            except Exception as e:
                st.error(f"Could not parse the table.\\nError: {str(e)}")
        else:
            st.warning("Please paste some table content first.")

    if st.session_state.df is not None:
        st.markdown("3. Edit values directly in the table below")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_config={
                "Test": st.column_config.TextColumn("Test name", required=True),
                "Result": st.column_config.NumberColumn("Result", min_value=0.0, step=0.01),
                "Unit": st.column_config.TextColumn("Unit"),
                "Reference Range": st.column_config.TextColumn("Reference range"),
                "Flag": st.column_config.SelectboxColumn(
                    "Flag",
                    options=["", "H", "L", "H*", "L*", "Critical", "Abnormal"],
                    required=False
                ),
            }
        )

        if st.button("4. Process edited table → Ready for questions", type="primary"):
            with st.spinner("Building vector index..."):
                # Convert table to text
                lines = ["Test | Result | Unit | Reference Range | Flag"]
                for _, row in edited_df.iterrows():
                    row_str = " | ".join(str(val) for val in row if pd.notna(val) and str(val).strip())
                    if row_str.strip():
                        lines.append(row_str)

                full_text = "\\n".join(lines)

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(full_text)
                docs = [Document(page_content=ch) for ch in chunks]

                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                # Prompt
                prompt_template = """You are a careful lab report assistant.
Use ONLY the information from the report table excerpts below.
If a value is missing or normal → say "not found in report" or "within normal range".
Never diagnose diseases. Only report values, flags, ranges.
Report table excerpts:
{context}
Question: {input}
Answer (concise, factual, include unit/range/flag):"""
                prompt = ChatPromptTemplate.from_template(prompt_template)

                # Groq LLM wrapper (LangChain compatible)
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.15,
                    max_tokens=1200,
                    api_key=st.session_state.groq_api_key
                )

                qa_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, qa_chain)
                
                st.session_state.rag_chain = rag_chain

            st.success(f"Table processed! ({len(chunks)} chunks) → Ask questions now.")

    # ── Chat area ────────────────────────────────────────
    if st.session_state.rag_chain is not None:
        st.divider()
        st.markdown("### Ask questions about the current report")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Ask anything about the report (e.g. 'Is glucose high?')"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    try:
                        response = st.session_state.rag_chain.invoke({"input": query})
                        answer = response["answer"].strip()
                        st.markdown(answer)
                        elapsed = time.time() - start_time
                        st.caption(f"Answered in {elapsed:.1f} seconds")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        answer = f"Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # ── Download Q&A ────────────────────────────────────────────────
        if st.session_state.messages:
            st.markdown("---")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            md_content = "# Blood Report Q&A\\n"
            md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    md_content += f"**You:**\\n{msg['content']}\\n\\n"
                else:
                    md_content += f"**Assistant:**\\n{msg['content']}\\n\\n"
                    md_content += "---\\n\\n"

            # Download button
            st.download_button(
                label="📥 Download this Q&A conversation",
                data=md_content,
                file_name=f"blood_report_qa_{timestamp}.md",
                mime="text/markdown",
                help="Saves all questions and answers in nicely formatted markdown",
                use_container_width=False
            )

        # ── Recommendation interface ───────────────────────────────────────────
        if st.session_state.rag_chain is not None:
            st.divider()
            st.subheader("General Recommendations (not medical advice)")

            if st.button("Get Recommendations for Abnormal Values", type="primary", use_container_width=True):
                with st.spinner("Generating general suggestions..."):
                    # Safety check: make sure API key exists
                    if "groq_api_key" not in st.session_state or not st.session_state.groq_api_key:
                        st.error("Groq API key is missing or invalid. Please set it again in the sidebar.")
                        st.stop()

                    # Use the same retriever to get context (abnormal values)
                    abnormal_context = st.session_state.rag_chain.invoke({"input": "any abnormal report"})["answer"].strip()

                    # New prompt for recommendations
                    rec_prompt_template = """You are a general health information assistant.
Based on the abnormal lab values below, provide ONLY very general suggestions for recovery.
For each abnormal value:
- Suggest common lifestyle, diet changes (e.g. exercise, low sugar diet)
- Mention general medicine classes if relevant (e.g. "doctors may consider statins for high cholesterol")
- ALWAYS say: "This is not medical advice. Consult a qualified doctor for personalized treatment and medicines."
- NEVER prescribe specific medicines or dosages.
- NEVER diagnose diseases.

Abnormal values from report:
{abnormal_context}

Answer in bullet points, be concise and cautious."""

                    rec_prompt = ChatPromptTemplate.from_template(rec_prompt_template)

                    # Use same LLM — with safety
                    rec_llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2,
                        max_tokens=800,
                        api_key=st.session_state.groq_api_key
                    )

                    # Simple chain for recommendations
                    rec_chain = rec_prompt | rec_llm

                    try:
                        rec_response = rec_chain.invoke({"abnormal_context": abnormal_context})
                        rec_answer = rec_response.content.strip()
                        st.markdown(rec_answer)
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")

            st.caption("These are general ideas only. Always see a doctor for real advice.")

with tab2:
    st.markdown("""
    ### How to use:
    1. **Paste** blood test report (PDF/Excel/WhatsApp)
    2. **Parse** → Edit values in table  
    3. **Process** → AI analyzes + saves to TiDB
    4. **Ask** questions about your results
    5. **Download** Q&A session
    
    ### Your TiDB database receives:
    ```sql
    INSERT INTO blood_reports (timestamp, test_name, result, unit, ref_range, flag)
    ```
    """)
