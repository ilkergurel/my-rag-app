"""
Reading and splitting documents
"""
import torch
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from collections import defaultdict
import chromadb
from chromadb.config import Settings
from chromadb import HttpClient
from datetime import datetime
from tqdm import tqdm
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import os
import warnings 
import logging
import logging.handlers
logging.getLogger("chromadb").setLevel(logging.WARNING)


warnings.filterwarnings('ignore') 
import numpy as np
import os
from multiprocessing import Process, Value, Lock,  Manager, JoinableQueue
from threading import Thread
from dotenv import load_dotenv

load_dotenv()


def collect_files(directory):
    """
    First get the file names and labels from file, then traverse the given directory and collect all file paths and put labels.
    """
    file_dict = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf') or file.lower().endswith('.epub'):      # Check if it's a .pdf or .epub file  
                file_path = os.path.join(root, file)                                 #Obtain filepath and remove path length control with \\?\\
                file_name, file_ext = os.path.splitext(file)                         #Obtain filename without extension                                   
                parent_path = os.path.basename(os.path.dirname(file_path))           #Obtain parent directory
                file_size = os.path.getsize(file_path)                               #Obtain file size
                file_dict[file_name].append([file_path, parent_path, file_size, file_ext]) 

    return file_dict

def create_batches(data, ids, batch_size):
    """Yield successive batch_size-sized chunks from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], ids[i:i + batch_size] 


def process_file(input_file, collection, embedding_function, file_counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files):
    #Initialization

    #embedding = OllamaEmbeddings(model="nomic-embed-text")
    embedding =OllamaEmbeddings(model="bge-m3")

    #chroma run --host localhost --port 8000 --path D:/Bilgi/__Databases/Bilgi_my_chroma_db_with_langchain3
    
    # Initialize Chroma client with persistence
    client = chromadb.HttpClient(
        host="localhost",  # Server's hostname or IP address
        port=8000,         # Port number the server is listening on
        settings=Settings(),
    )
    collection = client.get_or_create_collection(name="my-rag-db")    

 
    #persist_directory = "D:\\my_chroma_db_with_langchain"
    #client = chromadb.PersistentClient(path=persist_directory)
    #collection = client.get_or_create_collection(name="my-rag-db")


    for file, info  in input_file.items():
        for info_line in info:
        
            file_path = info_line[0]
            file_type = info_line[3]
            file_size = info_line[2]


            warning_flag = False
            error_flag = False
            if file_type.lower() == ".pdf":
                try:
                    loader = PyPDFLoader(file_path, mode="single")
                    doc = loader.load()      
                except Warning as w:
                    warning_flag = True
                    #warning_logger.warning(f"{file_counter} -- {file_path}: {w}")
                except Exception as e:
                    # Capture the exception and traceback
                    error_flag = True      
                    #error_logger.error(f"{file_counter} -- {file_path}: {e}", exc_info=True)
                    
            else:
                if file_type.lower() == ".epub":
                    try:
                        loader = UnstructuredEPubLoader(file_path)
                        doc = loader.load()     
                    except Warning as w:
                        warning_flag = True
                        #warning_logger.warning(f"{file_counter} -- {file_path}: {w}")
                    except Exception as e:
                        # Capture the exception and traceback
                        error_flag = True      
                        #error_logger.error(f"{file_counter} -- {file_path}: {e}", exc_info=True)  

            with lock:
                file_counter.value += 1  
                if error_flag:
                    shared_err_list.append(file_path)
                if warning_flag:
                    shared_warn_list.append(file_path)
            

            if (not error_flag) and (not warning_flag): 
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512, chunk_overlap=128, add_start_index=True
                )
                chunks = text_splitter.split_documents(doc)

                with lock:
                    id_counter.value += 1  

                print(f"Processed file no: {str(file_counter.value)} / {str(id_counter.value)} / {str(number_of_files)} - Adding file: {file_path}")

                # Generate unique IDs for each chunk
                chunk_ids = [f"doc_{id_counter.value}_chunk_{i}" for i in range(len(chunks))]
                
                batch_size = 128  # Adjust based on your system's capacity
                total_chunks = len(chunks)
                                


                #Add chunks data in batches for speed consideration. Operation is same as one chunk at a time in chromadb.
                #with tqdm(total=total_chunks, desc=f"{str(file_counter.value)} / {str(number_of_files)} - Adding chunks of {file_path}", position=1, leave=True, unit="chunks") as pbar:        
                for batch, batch_ids in create_batches(chunks, chunk_ids, batch_size):
                    # Precompute embeddings for the current batch
                    batch_documents = [chunk.page_content.encode('utf-8', errors='replace').decode('utf-8') for chunk in batch]
                    batch_metadata = [chunk.metadata for chunk in batch]
                    batch_embeddings = embedding.embed_documents(batch_documents)

                    #Add the batch to the collection
                    collection.upsert(
                        documents=batch_documents,
                        metadatas=batch_metadata,
                        ids=batch_ids,
                        embeddings=batch_embeddings
                    )

                #print(f"Processed file no: {str(file_counter.value)} / {str(id_counter.value)} / {str(number_of_files)} - # of chunks: {total_chunks} - Adding chunks of {file_path}")
                    
                        # Update the progress bar
                        #pbar.update(len(batch))
            




def worker(file_queue, collection, embedding_function, file_counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files):
    """
    Worker function to process files from the queue.
    """

    while True:
        input_file = file_queue.get()
        if input_file is None:  # Sentinel value to signal the end
            file_queue.task_done()
            break
        process_file(input_file, collection, embedding_function, file_counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files)
        file_queue.task_done()


def process_worker(file_queue, collection, embedding_function, num_threads, counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files):
    """
    Function run by each process to start threads.
    """

    threads = []
    for _ in range(num_threads):
        t = Thread(target=worker, args=(file_queue, collection, embedding_function, counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()


def main(input_files, collection, embedding_function, num_processes):
    """
    Main function to set up multiprocessing and multithreading.

    """
    manager = Manager()

    # Create a queue to hold the file paths
    file_queue = JoinableQueue()
    
    # Populate the queue with file paths
    for file_path, info in input_files.items():
        file_queue.put({file_path : info})

    for _ in range(num_processes):
        file_queue.put(None)        
    
    # Total number of files
    number_of_files = len(input_files)
    
    # Create a shared counter and a lock
    counter = Value('i', 0)
    id_counter = Value('i', 0)
    shared_warn_list = manager.list()
    shared_err_list = manager.list()
    lock = Lock()
    
    # Create and start processes
    processes = []
    for _ in range(num_processes):
        p = Process(target=worker, args=(file_queue, collection, embedding_function, counter, id_counter, shared_warn_list, shared_err_list, lock, number_of_files))
        p.start()                             
        processes.append(p)
    
    # Wait for all files to be processed
    file_queue.join()

    with open("error_files_list.txt", "w", encoding='utf-8') as f1:
        f1.write(str(shared_err_list))
    with open("warning_files_list.txt", "w", encoding='utf-8') as f2:
        f2.write(str(shared_warn_list))  
    
    # Stop the worker processes
    for p in processes:
        p.join()
        print(f"Process {p.pid} exited with code {p.exitcode}")

      


if __name__ == "__main__":

    pdf_directory = "D:\\Bilgi"

    # Recursively find all .pdf files in subdirectories

    input_files = collect_files(pdf_directory)

    total_num_documents = len(input_files)
    print(f"Total number of documents (pdf + epub): {total_num_documents}")

    collection = 0

    print("Vectorizing process starts...")
    num_processes = 12 #int(os.cpu_count() / 2)  # Number of processes to match CPU cores

    print(f"Number of processes: {num_processes} for number of docs: {total_num_documents }")

            # Initialize the embedding function
    embedding_function = 0

    main(input_files, collection, embedding_function,num_processes)

