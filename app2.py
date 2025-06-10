import pandas as pd
import asyncio
import time
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load LLM
llm = Ollama(model="mistral")

# Prompt Template for final answer
qa_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""
You are an expert at reading embassy websites.

Use the content below to answer the question.

CONTENT:
{text}

QUESTION:
{question}

ANSWER:
"""
)

# Read URLs from Excel
def read_urls_from_excel(file_path):
    df = pd.read_excel(file_path)
    urls = df["URL"].dropna().unique().tolist()
    return urls, df

# Write extracted results to Excel
def write_results_to_excel(df, results_dict, output_path):
    df['Address'] = df['URL'].map(lambda url: results_dict.get(url, {}).get("Address", "Not Found"))
    df['Phone'] = df['URL'].map(lambda url: results_dict.get(url, {}).get("Phone", "Not Found"))
    df['Email'] = df['URL'].map(lambda url: results_dict.get(url, {}).get("Email", "Not Found"))
    df['OfficeHours'] = df['URL'].map(lambda url: results_dict.get(url, {}).get("OfficeHours", "Not Found"))
    df.to_excel(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

# Split text into chunks
def split_text_into_chunks(pages_data, chunk_size=500, chunk_overlap=50):
    all_text = "\n\n".join([text for _, _, text in pages_data])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(all_text)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

# Embed and store chunks into FAISS
def embed_and_store_chunks(chunks):
    print("🧠 Generating embeddings for chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    print("✅ Stored in FAISS vector store.")
    return vectorstore

# Retrieve relevant chunks
def retrieve_relevant_chunks(vectorstore, question, k=70):
    print(f"🔎 Retrieving top {k} chunks for: '{question}'")
    return vectorstore.similarity_search(question, k=k)

# def extractehhei=kr ki

# Ask LLM the final question
def ask_llm_about_embassy(llm, retrieved_chunks, question):
    combined_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    chain = LLMChain(llm=llm, prompt=qa_prompt)
    print("🧠 Thinking...")
    result = chain.invoke({"text": combined_text, "question": question})
    print(result)
    return result

# Extract fields from LLM response using simple parsing
# def extract_field(field_name, llm_response):
#     text = llm_response["text"] if isinstance(llm_response, dict) else llm_response
#     lines = text.splitlines()
#     for line in lines:
#         if field_name.lower() in line.lower():
#             return line.split(":", 1)[-1].strip()
#     return "Not Found"

# Crawl all pages (homepage + navbar)
async def crawl_all_pages_and_collect_text(url):
    all_pages_data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)

        try:
            homePage = await page.inner_text('body')
            all_pages_data.append(('Homepage', url, homePage))
        except:
            all_pages_data.append(('Homepage', url, "Could not find homepage"))

        link_elements = await page.query_selector_all("nav a , header a")
        navbar_links = []
        for elem in link_elements:
            href = await elem.get_attribute("href")
            if href and not href.startswith("javascript"):
                full_url = urljoin(url, href)
                navbar_links.append(full_url)
        navbar_links = list(set(navbar_links))

        for link in navbar_links:
            try:
                await page.goto(link, timeout=90000)
                time.sleep(1)
                text = await page.inner_text('body')
                all_pages_data.append(("NavbarPage", link, text))
            except Exception as e:
                all_pages_data.append(("NavbarPage", link, f"Failed to read: {e}"))

        await browser.close()
        return all_pages_data

# MAIN FLOW
file_path = "dataset (4).xlsx"  # Replace with actual uploaded file
urls, df = read_urls_from_excel(file_path)
print(df.head)
results = {}

import os

output_path = "final_output_file(152).xlsx"

# If the file doesn't exist, create it with headers
if not os.path.exists(output_path):
    df_blank = pd.DataFrame(columns=["URL", "Extracted_Info"])
    df_blank.to_excel(output_path, index=False)
    print("📝 Created new Excel file with headers.")

# Now inside the loop, append row-by-row
for url in urls:
    print(f"\n🌐 Processing: {url}")
    
    all_pages_data = asyncio.run(crawl_all_pages_and_collect_text(url))
    chunks = split_text_into_chunks(all_pages_data)
    vectorstore = embed_and_store_chunks(chunks) 
    
    question = "What is the physical address, telephone number, email ID, and office hours of this embassy?"
    retrieved_chunks = retrieve_relevant_chunks(vectorstore, question)
    llm_answer = ask_llm_about_embassy(llm, retrieved_chunks, question)
    llm_answer_text = llm_answer["text"]

    result_row = {
        "URL": url,
        "Address": llm_answer,#extract_field("Address", llm_answer),
        "Phone": llm_answer,#extract_field("Telephone", llm_answer),
        "Email": llm_answer,#extract_field("Email", llm_answer),
        "OfficeHours": llm_answer#extract_field("Office Hours", llm_answer)
    }

    # Load existing file and append the new row
    df_existing = pd.read_excel(output_path)
    df_updated = pd.concat([df_existing, pd.DataFrame([result_row])], ignore_index=True)
    df_updated.to_excel(output_path, index=False)
    print(f"✅ Saved data for {url}")