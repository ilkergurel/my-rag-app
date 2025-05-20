"""
Documentation sources:

https://api.python.langchain.com/en/v0.1/langchain_api_reference.html

https://python.langchain.com/docs/introduction/

https://langchain-ai.github.io/langgraph/

In order to download the web contents with all links, use the following command on cmd line for each source:
wget -r -A.html -P langgraph-docs https://langchain-ai.github.io/langgraph/

"""

"""
Reading and splitting documents
"""

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from collections import defaultdict
import chromadb
from chromadb.config import Settings
from chromadb import HttpClient
from datetime import datetime

import os
from markdown import markdown
from bs4 import BeautifulSoup, Comment
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
import warnings 
import logging
import logging.handlers
import time
import requests
import json
import msgspec

logging.getLogger("chromadb").setLevel(logging.WARNING)


warnings.filterwarnings('ignore') 

from multiprocessing import Process, Value, Lock,  Semaphore, Queue, JoinableQueue
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------------
class Data(msgspec.Struct):
    documents: list[str]
    metadata: list[dict]

# ---------------------------------------------------------------------------------
def collect_files(directory, extensions):
    """
    First get the file names and labels from file, then traverse the given directory and collect all file paths and put labels.
    """

    # Traverse the directory and collect file paths    
    file_counter=0  #it is required because some files may have the same name
    file_dict = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):                       
                file_path = os.path.join(root, file)                                 #Obtain filepath and remove path length control with \\?\\
                file_name, file_ext = os.path.splitext(file)                         #Obtain filename without extension                                   
                parent_path = os.path.basename(os.path.dirname(file_path))           #Obtain parent directory
                file_size = os.path.getsize(file_path)                               #Obtain file size
                file_dict[file_path].append([file_name, parent_path, file_size, file_ext])     
                
                file_counter += 1                
    
    print(f"Number of files under all folders (files may have same name): {file_counter}")

    return file_dict, file_counter

# ---------------------------------------------------------------------------------
def merge_json_files(input_folder, output_folder, file_name):
    """Merge multiple JSON files into a single JSON file."""
    all_documents = []
    all_metadata = []
    file_counter=0  #it is required because some files may have the same name

    for root, dirs, files in os.walk(input_folder):
        for file in files:     
            file_path = os.path.join(root, file)                                 
      
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)               
                    # Extend the lists with data from each file
                    all_documents.extend(data.get('documents', []))
                    all_metadata.extend(data.get('metadata', []))

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

            file_counter += 1

    # Combine the lists into a single dictionary
    merged_data = {
        'documents': all_documents,
        'metadata': all_metadata
    }

    with open(output_folder + file_name, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------------
def html_to_markdown(html_content, task_id, logger):
    """Convert HTML content to markdown"""

    # Parse HTML with BeautifulSoup (optional cleanup can be done here)
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.exception(f"Failed to parse HTML content: {e}")
        return None, None
        
    # Remove excluded tags (form, header, footer, nav)
    for tag in soup(['form', 'header', 'footer', 'nav', 'iframe', 'script', 'a', 'img']):
        tag.decompose()    

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Get plain text version
    text = soup.get_text('\n', strip=True)

    # #Initialize HTML2Text with your settings
    # h = html2text.HTML2Text()
    # h.ignore_links = True
    # h.ignore_images = True
    # h.skip_internal_links = True    

    # # Get cleaned HTML
    # cleaned_html = str(soup)
    
    # # Escape HTML if needed (for remaining tags you want to show as text)
    # cleaned_html = html.escape(cleaned_html)
    
    # # Convert to markdown
    # markdown = h.handle(cleaned_html)    

    # # Clean up excessive newlines
    # markdown = '\n'.join(line for line in markdown.split('\n') if line.strip())
    markdown = "0"

    return markdown, text  

# ---------------------------------------------------------------------------------
class RetryableOllamaEmbeddings:
    """Create Ollama embedding function which has retry capability in case of connection problem. """

    def __init__(self, logger, model: str = "bge-m3", base_url: str = "http://localhost:11434", retries: int = 5, delay: int = 1):
        self.embedding = OllamaEmbeddings(model=model, base_url=base_url)
        self.retries = retries
        self.delay = delay
        self.logger = logger

    def embed_documents(self, texts, task_id, logger):
        for attempt in range(1, self.retries + 1):
            try:
                return self.embedding.embed_documents(texts)
            except Exception as e:
                self.logger.exception(f"Attempt {attempt} failed: {e}")
                time.sleep(self.delay)
        logger.info("All retry attempts for embed_documents failed.")

    def embed_query(self, text):
        for attempt in range(1, self.retries + 1):
            try:
                return self.embedding.embed_query(text)
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(self.delay)
        raise Exception("All retry attempts for embed_query failed.")

# ---------------------------------------------------------------------------------
def get_doc_from_html(file_path, logger, file_no, file_path_print, task_id):
    """Read html documents, obtain text content and put them in document with metadata"""

    #Initialize output variables    
    error_flag = False         
    html_content = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError as e:
        logger.error(f"File not found - file_counter:{file_no} -- {file_path_print}")

    text = None
    markdown = None                  
    try:    
        markdown, text = html_to_markdown(html_content, task_id, logger)
    except Exception as e:
        # Capture the exception and traceback
        error_flag = True  
        logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")        

    doc = [Document(
        page_content=text if text is not None else "",
        metadata={"source": file_path_print}
    )]  

    return doc, error_flag

# ---------------------------------------------------------------------------------
def get_doc_from_pdf_epub(file_path, logger, file_no, file_path_print, task_id):
    """Read pdf and epub documents, obtain text content and put them in document with metadata"""

    #Initialize output variables
    error_flag = False  
    doc = [Document(
        page_content="",
        metadata={"source": ""}
    )] 

    if file_path.lower().endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path, mode="single")
            doc = loader.load()      
        except Exception as e:
            # Capture the exception and traceback
            error_flag = True      
            logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")
            
    elif file_path.lower().endswith(".epub"):
            try:
                loader = UnstructuredEPubLoader(file_path)
                doc = loader.load()     
            except Exception as e:
                # Capture the exception and traceback
                error_flag = True      
                logger.exception(f"Error in task {task_id}: {e} - file_counter: {file_no} -- {file_path_print}") 
    
    return doc, error_flag

# ---------------------------------------------------------------------------------
def create_batches(data, ids, batch_size):
    """Yield successive batch_size-sized chunks from data."""

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], ids[i:i + batch_size] 

# ---------------------------------------------------------------------------------
def process_file(input_file, extensions, semaphore, client_chromadb, embedding_function, file_counter, lock, number_of_files, logger, task_id):
    """
    Read files, split them, embed them and put the embedded chunks to database
    """

    #This process deals with a single file. So, file counter increments once.
    with lock:
        file_counter.value += 1  
        file_no = file_counter.value   


    collection = None
    try:
        collection = client_chromadb.get_or_create_collection(name="my-doc-assistant-db")   
    except Exception as e:
        logger.exception(f"ChromaDB error - {task_id}: {e}")


    #persist_directory = "D:\\my_chroma_db_with_langchain"
    #client = chromadb.PersistentClient(path=persist_directory)
    #collection = client.get_or_create_collection(name="my-rag-db")


    for file_path, info  in input_file.items():
        #Obtain file path + name and file extension
        file_path_print = file_path[4:]  #From file path "\\\\?\\" part is removed
        file_extension = info[0][3]

        #Inintialize inputs
        error_flag = False  
        doc = [Document(
            page_content="",
            metadata={"source": ""}
        )] 

        if file_extension == ".html" or file_extension == ".htm":
            doc, error_flag = get_doc_from_html(file_path, logger, file_no, file_path_print, task_id)
        elif file_extension == ".pdf" or file_extension == ".epub":
            doc, error_flag = get_doc_from_pdf_epub(file_path, logger, file_no, file_path_print, task_id)

        #If no error and non-empty input text, continue processing of the current file    
        if not error_flag and (doc[0].page_content is not None) and (len(doc[0].page_content) > 0): 

            #Start text splitting into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=128, add_start_index=True
            )
            chunks = text_splitter.split_documents(doc) 
            

            print(f"Processed file no: {str(file_no)} / {str(number_of_files)} - len(chunks): {len(chunks)} - Adding file: {file_path_print}")


            #Create bm25 retriever and save it for one time
            data_to_save = {
                'documents': [chunk.page_content.encode('utf-8', errors='replace').decode('utf-8') for chunk in chunks],
                'metadata': [{"source": chunk.metadata["source"]} for chunk in chunks]
            }
            
            #For bm25 retriever, save the chunks obtained from one file/document for one time
            try:
                folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db-temp\\"
                with open(folder +f"bm25_data_docs_and_metadata_temp_{file_no}.json", 'w', encoding='utf-8') as file:   
                    json.dump(data_to_save, file, ensure_ascii=False, indent=4)   
            except Exception as e:
                logger.exception(f"Error in task {task_id} in json.dump: {e} - file_counter: {file_no} -- {file_path_print}")   


            # From this part on, embedded vector data is created and saved to database
            # Generate unique IDs for each chunk
            chunk_ids = [f"doc_{file_no}_chunk_{i}" for i in range(len(chunks))]
            
            batch_size = 128  # Adjust based on your system's capacity
                            
            #Add chunks data in batches for speed consideration. Operation is same as one chunk at a time in chromadb.      
            for batch, batch_ids in create_batches(chunks, chunk_ids, batch_size):
                # Precompute embeddings for the current batch
                batch_documents = [chunk.page_content.encode('utf-8', errors='replace').decode('utf-8') for chunk in batch]
                batch_metadata = [chunk.metadata for chunk in batch]

                batch_embeddings = []
                try:
                    with semaphore:
                        batch_embeddings = embedding_function.embed_documents(batch_documents, task_id, logger)
                except Exception as e:
                    logger.exception(f"Embedding error - {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")        


                try:
                    #time.sleep(random.uniform(0.1, 0.3))
                    if collection is not None:
                        collection.upsert(
                            documents=batch_documents,
                            metadatas=batch_metadata,
                            ids=batch_ids,
                            embeddings=batch_embeddings
                        )
                    else:
                        logger.error(f"Collection is not initialized - {task_id}: file_counter: {file_no} -- {file_path_print}")
                except Exception as e:
                    logger.exception(f"ChromaDB upsert error - {task_id}: {e} - file_counter: {file_no} -- {file_path_print}")

        elif doc[0].page_content is None or len(doc[0].page_content) == 0:
            logger.info(f"html file has no text. So, there is no embedded vector - {task_id}: file_counter: {file_no} -- {file_path_print}")

# ---------------------------------------------------------------------------------
def worker(file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files):
    """
    Each worker is run under a different process in multiprocessing case
    Worker function to process files from the queue.
    Logger created, embedding function created, chromadb client created
    Chromadb server should be started with following command on powershell:
        chroma run --host localhost --port 8000 --path D:/Langchain-Langgraph-Doc-WebSites/__Databases/langchain-docs-vectordb
    """

    #Create logger
    logger = logging.getLogger(f'Worker-{task_id}')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    #For multiprocessing, create the embedding function having retry mechanism for Ollama embeddings
    embedding_function = RetryableOllamaEmbeddings(logger, model="bge-m3", retries=5, delay=1)

    # Initialize Chroma client with persistence
    client_chromadb = chromadb.HttpClient(
        host="localhost",  # Server's hostname or IP address
        port=8000,         # Port number the server is listening on
        settings=Settings(),
    )    

    while True:
        input_file = file_queue.get()
        if input_file is None:  # Sentinel value to signal the end
            file_queue.task_done()
            break
        process_file(
            input_file,
            extensions, 
            semaphore, 
            client_chromadb, 
            embedding_function, 
            file_counter, 
            lock, 
            number_of_files,
            logger,
            task_id)
        
        file_queue.task_done()

# ---------------------------------------------------------------------------------
def process_worker(num_threads, file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files):
    """
    Function run by each process to start threads.
    Not utilized now !!!
    """

    #Create logger
    logger = logging.getLogger(f'Worker-{task_id}')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    threads = []
    for _ in range(num_threads):
        t = Thread(target=worker, args=(file_queue, extensions, semaphore, file_counter, task_id, log_queue, lock, number_of_files))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

# ---------------------------------------------------------------------------------
def configure_listener():
    """
    Create logger. Configures handlers for the listener.
    """
    # File handler
    file_handler = logging.FileHandler('rag_multiprocessing.log')
    file_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)

    ##Console handler
    # console_handler = logging.StreamHandler()
    # console_formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
    # console_handler.setFormatter(console_formatter)
    #return [file_handler, console_handler]
    return [file_handler]

# ---------------------------------------------------------------------------------
def main():
    """
    Main function to set up multiprocessing and multithreading.

    """
    files_folder = '\\\\?\\D:\\Langchain-Langgraph-Doc-WebSites\\langchain-langgraph-docs'

    num_processes = 8 #int(os.cpu_count() / 2)  # Number of processes to match CPU cores
    max_concurrent_calls = 8  # Considering Ollama simultaneous call with retries
    semaphore = Semaphore(max_concurrent_calls)
    
    extensions=("html", "htm", "pdf", "epub")  # Specify the file extensions to look for
    input_files, number_of_files = collect_files(files_folder, extensions)

    print("Vectorizing process starts...")

    print(f"Number of processes: {num_processes}")

    # Create a queue to hold the file paths
    file_queue = JoinableQueue()

    #Create a separate queue for logging
    log_queue = JoinableQueue()
    # Configure handlers for the listener
    handlers = configure_listener()
    # Set up the QueueListener with the handlers
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()    


    logger = logging.getLogger(f'main')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)


    # Populate the queue with file paths
    for file_path, info in input_files.items():
        file_queue.put({file_path : info})

    for _ in range(num_processes):
        file_queue.put(None)        
        
    # Create a shared counter and a lock
    counter = Value('i', 0)
    lock = Lock()
    
    # Create and start processes
    processes = []
    for task_no in range(num_processes):
        p = Process(target=worker, args=(
            file_queue, extensions, semaphore, counter, task_no, log_queue, lock, number_of_files
            ))
        p.start()                             
        processes.append(p)
    
    while processes:
        for task_no, p in enumerate(processes):
            p.join(timeout=0.1)  # Non-blocking join
            if not p.is_alive():
                if p.exitcode == 0:
                    print(f"Process {p.pid} completed successfully with exit code {p.exitcode}")
                    logger.info(f"Process {p.pid} completed successfully with exit code {p.exitcode}")
                    processes.pop(task_no)
                else:
                    print(f"Process {p.pid} crashed with exit code {p.exitcode}. Restarting...")
                    logger.info(f"Process {p.pid} crashed with exit code {p.exitcode}. Restarting...")                            
                    new_p = Process(target=worker, args=(
                        file_queue, extensions, semaphore, counter, task_no, log_queue, lock, number_of_files
                        ))
                    new_p.start()                       
                    processes[task_no] = new_p
        time.sleep(1)        

    # Wait for all files to be processed
    file_queue.join()
    print("file_queue is empty. ")


    input_folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db-temp\\"
    output_folder = "D:\\Langchain-Langgraph-Doc-WebSites\\__Databases\\langchain-docs-bm25-db\\"
    file_name = "bm25_data_docs_and_metadata.json"
    try:
        merge_json_files(input_folder, output_folder, file_name)
        print("All json temp files merged...")
    except Exception as e:
        logger.exception(f"merge_json_files() function error.")        


    # Stop the listener
    listener.stop()    
    print("listener stopped...")  

    print("All processes finished...")  


if __name__ == "__main__":
    main()

