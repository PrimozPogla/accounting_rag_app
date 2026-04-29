import os
import re
import hashlib
import numpy as np
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


st.set_page_config(
    page_title="Accounting RAG Assistant",
    page_icon="A",
    layout="wide"
)


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    padding: 2.4rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
}

.hero h1 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: 0.4rem;
}

.hero p {
    color: #dbeafe;
    font-size: 1.05rem;
}

.card {
    background: white;
    padding: 1.4rem;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

.answer-card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    margin-top: 1rem;
}

.metric-box {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    text-align: center;
}

.metric-number {
    font-size: 1.45rem;
    font-weight: 800;
    color: #1d4ed8;
}

.metric-label {
    color: #64748b;
    font-size: 0.86rem;
}

.meta-pill {
    display: inline-block;
    background: #e0ecff;
    color: #1e40af;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.82rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-weight: 600;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}

.small-muted {
    color: #64748b;
    font-size: 0.92rem;
}

.stTextInput > div > div > input {
    border-radius: 14px;
    padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


class LocalEmbeddings(Embeddings):
    """
    Lightweight local embeddings without sentence-transformers / torch.
    This avoids heavy dependencies and works better on Streamlit Cloud.
    """

    def __init__(self, dim=384):
        self.dim = dim

    def _embed(self, text):
        vector = np.zeros(self.dim)

        words = re.findall(r"\b\w+\b", text.lower())

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            index = h % self.dim
            vector[index] += 1

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def load_documents(folder_path="documents"):
    docs = []

    if not os.path.exists(folder_path):
        st.error("The documents folder was not found.")
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(
                os.path.join(folder_path, file),
                encoding="utf-8"
            )
            docs.extend(loader.load())

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


@st.cache_resource
def create_vectorstore():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = LocalEmbeddings()

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings
    )

    return db, chunks, documents


def parse_query(query):
    q = query.lower()

    intent = {
        "object": None,
        "timing": None,
        "special": None,
        "question_type": "general"
    }

    if (
        "financial statements" in q
        or "effect" in q
        or "affect" in q
        or "impact" in q
        or "cash flow" in q
        or "balance sheet" in q
        or "income statement" in q
    ):
        intent["question_type"] = "financial_effect"

    elif (
        "journal entry" in q
        or "how to post" in q
        or "debit" in q
        or "credit" in q
        or "booking" in q
    ):
        intent["question_type"] = "journal_entry"

    elif (
        "mistake" in q
        or "wrong" in q
        or "incorrect" in q
        or "common error" in q
    ):
        intent["question_type"] = "common_mistakes"

    if "machine" in q or "equipment" in q or "fixed asset" in q or "vehicle" in q:
        intent["object"] = "fixed_asset"
    elif "goods" in q or "inventory" in q or "stock" in q:
        intent["object"] = "inventory"
    elif "rent" in q or "insurance" in q or "subscription" in q:
        intent["object"] = "service"

    if (
        "will pay later" in q
        or "pay later" in q
        or "paid later" in q
        or "deferred payment" in q
    ):
        intent["timing"] = "later"

    if (
        "paid immediately" in q
        or "pay immediately" in q
        or "cash purchase" in q
        or "paid now" in q
    ):
        intent["timing"] = "immediate"

    if (
        "paid before" in q
        or "pay before" in q
        or "before receiving" in q
        or "before delivery" in q
    ):
        intent["timing"] = "before"

    if (
        "customer paid before" in q
        or "customer pays before" in q
        or "customer deposit" in q
    ):
        intent["timing"] = "customer_before"

    if "customer paid invoice" in q or "customer paid" in q or "paid outstanding invoice" in q:
        intent["special"] = "customer_payment"

    if "supplier payment" in q or "paid supplier" in q or "pay supplier" in q:
        intent["special"] = "supplier_payment"

    if "depreciation" in q or "depreciate" in q:
        intent["special"] = "depreciation"

    if "damaged" in q or "obsolete" in q or "cannot be sold" in q or "write off" in q or "write-off" in q:
        intent["special"] = "inventory_writeoff"

    if "no invoice" in q or "not received the invoice" in q or "invoice not received" in q:
        intent["special"] = "accrued_expenses"

    if "used electricity" in q or "incurred expense" in q:
        intent["special"] = "accrued_expenses"

    if "paid rent" in q or "next 12 months" in q or "next months" in q or "insurance paid in advance" in q:
        intent["special"] = "prepaid_expenses"

    if "returned goods" in q or "sales return" in q or "credit note" in q or "price reduction" in q:
        intent["special"] = "credit_note"

    return intent


def map_intent_to_file(intent):
    if intent["special"] == "customer_payment":
        return "customer_payment_received"
    if intent["special"] == "supplier_payment":
        return "supplier_payment"
    if intent["special"] == "depreciation":
        return "depreciation_of_fixed_assets"
    if intent["special"] == "inventory_writeoff":
        return "inventory_write-off"
    if intent["special"] == "accrued_expenses":
        return "accrued_expenses"
    if intent["special"] == "prepaid_expenses":
        return "prepaid_expenses"
    if intent["special"] == "credit_note":
        return "credit_note"

    if intent["timing"] == "customer_before":
        return "advance_payment_from_customer"
    if intent["timing"] == "before":
        return "advance_payment_to_supplier"

    if intent["object"] == "fixed_asset":
        if intent["timing"] == "later":
            return "purchase_of_fixed_asset_on_credit"
        if intent["timing"] == "immediate":
            return "purchase_of_fixed_asset_paid_immediately"

    if intent["object"] == "inventory":
        if intent["timing"] == "later":
            return "purchase_of_goods_on_credit"
        if intent["timing"] == "immediate":
            return "purchase_of_goods_paid_immediately"

    return None


def find_full_document_by_file(documents, target_file):
    for doc in documents:
        source = doc.metadata.get("source", "").lower().replace("\\", "/")

        if target_file in source:
            return doc

    return None


def find_full_document_from_semantic_result(documents, semantic_doc):
    semantic_source = semantic_doc.metadata.get("source", "").lower().replace("\\", "/")

    for doc in documents:
        source = doc.metadata.get("source", "").lower().replace("\\", "/")

        if source == semantic_source:
            return doc

    return semantic_doc


def extract_section(text, start_heading, end_headings):
    lower_text = text.lower()
    start = lower_text.find(start_heading.lower())

    if start == -1:
        return None

    end = len(text)

    for heading in end_headings:
        pos = lower_text.find(heading.lower(), start + len(start_heading))

        if pos != -1:
            end = min(end, pos)

    return text[start:end].strip()


def get_relevant_answer(text, question_type):
    if question_type == "financial_effect":
        section = extract_section(
            text,
            "EFFECT ON FINANCIAL STATEMENTS",
            ["Cash Flow Classification", "Journal Entry", "Example", "Common Mistakes", "Related Concepts"]
        )

        if section:
            return section

    if question_type == "journal_entry":
        section = extract_section(
            text,
            "Journal Entry",
            ["Example", "Common Mistakes", "Related Concepts"]
        )

        if section:
            return section

    if question_type == "common_mistakes":
        section = extract_section(
            text,
            "Common Mistakes",
            ["EFFECT ON FINANCIAL STATEMENTS", "Journal Entry", "Example", "Related Concepts"]
        )

        if section:
            return section

    return text


def format_answer_for_markdown(answer_text):
    text = answer_text.strip()

    text = text.replace("EFFECT ON FINANCIAL STATEMENTS", "### EFFECT ON FINANCIAL STATEMENTS")
    text = text.replace("Journal Entry", "### Journal Entry")
    text = text.replace("Common Mistakes", "### Common Mistakes")
    text = text.replace("Cash Flow Classification", "### Cash Flow Classification")

    text = re.sub(r"\nBalance Sheet:", "\n\n**Balance Sheet:**", text)
    text = re.sub(r"\nIncome Statement:", "\n\n**Income Statement:**", text)
    text = re.sub(r"\nCash Flow:", "\n\n**Cash Flow:**", text)

    text = re.sub(r"\nAt purchase:", "\n\n**At purchase:**", text)
    text = re.sub(r"\nAt payment:", "\n\n**At payment:**", text)
    text = re.sub(r"\nAt sale:", "\n\n**At sale:**", text)
    text = re.sub(r"\nAt delivery:", "\n\n**At delivery:**", text)
    text = re.sub(r"\nAt advance receipt:", "\n\n**At advance receipt:**", text)
    text = re.sub(r"\nAt advance payment:", "\n\n**At advance payment:**", text)
    text = re.sub(r"\nOver time:", "\n\n**Over time:**", text)

    return text


db, chunks, documents = create_vectorstore()


with st.sidebar:
    st.markdown("## Navigation")

    page = st.radio(
        "Go to",
        ["Home", "Search", "About / Statistics"],
        label_visibility="collapsed"
    )

    st.markdown("## Project Info")

    st.markdown(
        f"""
        <div class="card">
            <p><b>Documents:</b> {len(documents)}</p>
            <p><b>Chunks:</b> {len(chunks)}</p>
            <p><b>Embedding model:</b><br> Lightweight hash embeddings</p>
            <p><b>Chunk size:</b> 1200</p>
            <p><b>Chunk overlap:</b> 200</p>
            <p><b>Retrieval:</b><br> Rule-based parser + semantic fallback</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Accounting RAG Assistant</h1>
            <p>
                A scenario-based Retrieval-Augmented Generation application for accounting transaction guidance.
                The app searches a structured knowledge base of accounting examples and returns the most relevant accounting treatment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>1. Structured Documents</h3>
            <p>The knowledge base contains accounting scenarios such as purchases, sales, advances, depreciation, accruals, and write-offs.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>2. Lightweight Retrieval</h3>
            <p>User questions are converted into lightweight hash-based vectors and compared with document chunks.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>3. Accounting Guidance</h3>
            <p>The app retrieves relevant sections such as financial statement effects, journal entries, and common mistakes.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Example questions")

    st.markdown("""
    - We bought a machine and will pay later. What financial statements does it affect?
    - Customer paid invoice.
    - Does depreciation affect cash flow?
    - Goods are damaged and cannot be sold.
    - We paid before receiving goods.
    """)


elif page == "Search":
    st.markdown(
        """
        <div class="hero">
            <h1>Search Accounting Scenarios</h1>
            <p>
                Ask a natural-language question and retrieve the most relevant accounting treatment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-title'>Ask an accounting question</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Ask your question",
        placeholder="Example: We bought a machine and will pay later..."
    )

    if query:
        intent = parse_query(query)
        target_file = map_intent_to_file(intent)

        selected_doc = None
        retrieval_method = ""

        if target_file:
            selected_doc = find_full_document_by_file(documents, target_file)
            retrieval_method = f"Rule-based match: {target_file}"

        if selected_doc is None:
            semantic_results = db.similarity_search(query, k=1)

            if semantic_results:
                selected_doc = find_full_document_from_semantic_result(documents, semantic_results[0])
                retrieval_method = "Semantic search fallback"
            else:
                selected_doc = documents[0]
                retrieval_method = "Fallback to first document"

        answer_text = get_relevant_answer(
            selected_doc.page_content,
            intent["question_type"]
        )

        formatted_answer = format_answer_for_markdown(answer_text)

        st.markdown("<div class='section-title'>Answer</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <span class="meta-pill">{retrieval_method}</span>
            <span class="meta-pill">Question type: {intent['question_type']}</span>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown(formatted_answer)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Debug / Source"):
            st.write("Detected intent:")
            st.json(intent)

            st.write("Selected source:")
            st.write(selected_doc.metadata.get("source", "Unknown"))

            st.write("Full selected document:")
            st.write(selected_doc.page_content)

    else:
        st.markdown(
            """
            <div class="card">
                <b>Try one of these questions:</b><br><br>
                • We bought a machine and will pay later. What financial statements does it affect?<br>
                • Customer paid invoice.<br>
                • We paid before receiving goods.<br>
                • Does depreciation affect cash flow?<br>
                • Customer returned goods after invoice.
            </div>
            """,
            unsafe_allow_html=True
        )


elif page == "About / Statistics":
    st.markdown(
        """
        <div class="hero">
            <h1>About the Project</h1>
            <p>
                Technical overview of the RAG architecture, document base, chunking strategy, and retrieval approach.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-number'>{len(documents)}</div><div class='metric-label'>Documents</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-number'>{len(chunks)}</div><div class='metric-label'>Chunks</div></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            "<div class='metric-box'><div class='metric-number'>1200</div><div class='metric-label'>Chunk Size</div></div>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            "<div class='metric-box'><div class='metric-number'>200</div><div class='metric-label'>Chunk Overlap</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### Technical Details")

    st.markdown("""
    <div class="card">
        <b>Embedding model:</b> Lightweight hash embeddings<br>
        <b>Vector database:</b> ChromaDB<br>
        <b>Text splitter:</b> RecursiveCharacterTextSplitter<br>
        <b>Framework:</b> Streamlit<br>
        <b>Retrieval approach:</b> lightweight vector search with rule-based parsing for accounting-specific timing logic
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Chunking Strategy")

    st.markdown("""
    <div class="card">
        The final version uses <b>chunk_size = 1200</b> and <b>chunk_overlap = 200</b>.
        This was selected because smaller chunks often returned only the beginning of a scenario,
        while larger chunks preserved the accounting logic, financial statement effects, and journal entries.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Document Sources")

    st.markdown("""
    <div class="card">
        The documents were self-written as scenario-based accounting guides.
        Each document follows a consistent structure: keywords, scenario description, accounting logic,
        financial statement effects, journal entry, example, and related concepts.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Documents Included")

    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        title = doc.page_content.split("\n")[0]
        st.markdown(f"- **{title}**  \n  `{source}`")