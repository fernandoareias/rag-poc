import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv(".env")

required_env_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]

missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
def main():
    # Configurar os embeddings
    try:
        print("Configurando embeddings AzureOpenAIEmbeddings...")
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            model_kwargs={"model": os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")}
        )
        print("Embeddings configurados com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar embeddings: {e}")
        return

    # Carregar o banco vetorial FAISS
    vector_db_path = "./faiss-db"
    try:
        print("Carregando banco vetorial FAISS...")
        db = FAISS.load_local(vector_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        print("Banco vetorial carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o banco vetorial: {e}")
        return

    # Configurar o modelo de linguagem
    try:
        print("Configurando o modelo de linguagem AzureChatOpenAI...")
        llm = AzureChatOpenAI(
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            verbose=False,
            temperature=0.3,
        )
        print("Modelo configurado com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar o modelo AzureChatOpenAI: {e}")
        return

    # Definir template de prompt
    PROMPT_TEMPLATE = """You are an AI Assistant. Given the following context:
    {context}

    Answer the following question:
    {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Configurar QA com base no retriever
    try:
        print("Configurando RetrievalQA...")
        retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=False,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        print("RetrievalQA configurado com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar RetrievalQA: {e}")
        return

    # Loop para o chatbot
    print("Chatbot is ready! Type your questions below:")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        try:
            response = qa.invoke({"query": question})
            print("\nAnswer:", response["result"])
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}")

if __name__ == "__main__":
    main()
