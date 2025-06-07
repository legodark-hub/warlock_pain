import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAR_INFO_PATH = "./data/character_info.txt"
WORLD_INFO_PATH = "./data/world_info.txt"
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

def initialize_retriever(k_results: int = 5):
    """
    Инициализирует и возвращает ретривер на основе ChromaDB.
    Создает базу данных, если она не существует, или загружает существующую.
    """
    logger.info(f"Инициализация эмбеддинг-модели: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
    logger.info("Эмбеддинг-модель инициализирована.")

    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logger.info(f"Загрузка существующей базы данных ChromaDB из '{PERSIST_DIRECTORY}'.")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
        )
        logger.info("Существующая база данных ChromaDB успешно загружена.")
    else:
        logger.info(f"Создание новой базы данных ChromaDB в '{PERSIST_DIRECTORY}'...")
        character_loader = TextLoader(CHAR_INFO_PATH, encoding="utf-8")
        world_loader = TextLoader(WORLD_INFO_PATH, encoding="utf-8")
        character_docs_initial = character_loader.load()
        world_docs_initial = world_loader.load()

        for doc in character_docs_initial:
            doc.metadata["source"] = "character_story"
        for doc in world_docs_initial:
            doc.metadata["source"] = "world_history"

        all_docs = character_docs_initial + world_docs_initial
        logger.info(f"Загружено {len(all_docs)} документов для индексации.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(all_docs)

        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
        )
        logger.info(f"Новая база данных ChromaDB создана и сохранена в '{PERSIST_DIRECTORY}'. Разделено на {len(splits)} чанков.")

    logger.info(f"Ретривер инициализирован с k={k_results}.")
    return vectorstore.as_retriever(search_kwargs={"k": k_results})
