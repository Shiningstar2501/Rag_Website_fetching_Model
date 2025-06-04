from playwright.async_api import async_playwright
import asyncio 
from urllib.parse import urljoin 
import time

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert at reading embassy websites.

From the content below, extract the complete **physical address** , **Telephone Number**, **Email ID**, **Office Hours** of the embassy.

If no address, telephone nummber, email id, office hours  is found, say "Not Found" in that column.

CONTENT:
{text}

ADDRESS:
Telephone Number:
Email ID:
Office Hours:


"""
)
chain = LLMChain(llm=llm, prompt=prompt)

def extract_address_from_pages(all_pages_data):
    combined_text = "\n\n".join([text for _, _, text in all_pages_data])
    print("🧠 Thinking... this may take a moment depending on your LLM model...")
    result = chain.invoke({"text": combined_text})
    return result


def split_text_into_chunks(pages_data, chunk_size=500, chunk_overlap=50):
    all_text = "\n\n".join([text for _, _, text in pages_data])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(all_text)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks



def embed_and_store_chunks(chunks):
    print("🧠 Generating embeddings for chunks...")
    
    # You can swap this with OllamaEmbeddings if you're using a local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    print("✅ Stored in FAISS vector store.")
    return vectorstore


def retrieve_relevant_chunks(vectorstore, question, k=5):
    print(f"🔎 Retrieving top {k} chunks for: '{question}'")
    relevant_chunks = vectorstore.similarity_search(question, k=k)
    return relevant_chunks

def ask_llm_about_embassy(llm, retrieved_chunks, question):
    combined_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

    prompt = PromptTemplate(
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

    chain = LLMChain(llm=llm, prompt=prompt)
    print("🧠 Thinking...")
    result = chain.invoke({"text": combined_text, "question": question})
    return result


all_pages_data = []
async def crawl_all_pages_and_collect_text(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url,timeout=60000)

        
        try:
            homePage = await page.inner_text('body')
            all_pages_data.append(('Homepage', url, homePage))
        except: 
            all_pages_data.append(('Homepage',url, "Could not find homepage"))
        
        link_elements= await page.query_selector_all("nav a , header a")
        navbar_links=[]
        for elem in link_elements:
            href = await elem.get_attribute("href")
            if href and not href.startswith("javascript"):
                full_url = urljoin(url, href)
                navbar_links.append(full_url)
                # navbar_links.append(href)
        navbar_links = list(set(navbar_links))

        for link in navbar_links:
            try:
                await page.goto(link, timeout=60000)
                time.sleep(1)
                text =await page.inner_text('body')
                all_pages_data.append(("NavbarPage", link, text))
            except Exception as e:
                all_pages_data.append(("NavbarPage", link, f"Failed to read:{e}"))

    await browser.close()
    return all_pages_data

# Example:


url = "https://www.nz.emb-japan.go.jp/itpr_en/consular_office.html"
async def main():
    text = await crawl_all_pages_and_collect_text(url)
    print(text)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
asyncio.run(main())
# address = extract_address_from_pages(all_pages_data)
chunks = split_text_into_chunks(all_pages_data) # step 1 of rag
vectorstore = embed_and_store_chunks(chunks) # step 2 of rag


llm = Ollama(model="mistral")  # Or use 'gemma' for faster

question = "What is the physical address and phone number of this embassy?"
retrieved_chunks = retrieve_relevant_chunks(vectorstore, question)
answer = ask_llm_about_embassy(llm, retrieved_chunks, question)

print("🏛️ Extracted Info:\n", answer)
# print("🏛️ Extracted Embassy Address:", address)