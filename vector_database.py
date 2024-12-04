import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv(".env")

required_env_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_DEPLOYMENT_EMBEDDINGS",
]

missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def create_vector_database(txt_path, output_dir="./faiss-db"):
    print(f"Carregando conteúdo de {txt_path}...")
    with open(txt_path, "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file if line.strip()]  # Remove linhas vazias

    # Verificar se o arquivo contém textos
    if not texts:
        raise ValueError(f"O arquivo {txt_path} está vazio ou não contém linhas processáveis.")
    print(f"Total de linhas processadas: {len(texts)}")

    print("Criando embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model="text-embedding-ada-002",  # Certifique-se de que é o modelo correto para embeddings
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    # Enviar textos em lotes pequenos para evitar erros
    chunk_size = 10
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    all_embeddings = []
    for chunk in chunked_texts:
        try:
            print(f"Processando lote de tamanho {len(chunk)}...")
            chunk_embeddings = embeddings.embed_documents(chunk)
            all_embeddings.extend(chunk_embeddings)
        except Exception as e:
            print(f"Erro ao processar lote: {e}")

    if not all_embeddings:
        raise ValueError("Falha ao gerar embeddings. Verifique sua configuração e dados.")

    print("Criando banco de dados vetorial FAISS...")
    db = FAISS.from_texts(texts=texts, embedding=embeddings)

    db.save_local(output_dir)
    print(f"Banco de dados vetorial salvo em {output_dir}")

if __name__ == "__main__":
    create_vector_database("final.txt")
