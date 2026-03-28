from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import traceback
import timeit
import os

# Langchain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFium2Loader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"



class QueryDoc(APIView):

    def post(self, request):
        try:
            reffiles = request.data.get('files')
            question = request.data.get('question')

            if not reffiles or not question:
                return Response({'error': 'files and question required'}, status=400)

            file_paths = reffiles.split(",")

            start = timeit.default_timer()

            all_documents = []

            # FIXED FILE LOADER (NO ERROR VERSION)
            for file_path in file_paths:
                file_path = file_path.strip()

                print("Checking:", file_path)
                print("Exists:", os.path.exists(file_path))

                if not os.path.exists(file_path):
                    print("File not found:", file_path)
                    continue

                try:
                    # ✅ PDF
                    if file_path.endswith(".pdf"):
                        try:
                            loader = PyPDFium2Loader(file_path)
                            docs = loader.load()
                        except Exception as e:
                            print("PyPDFium2 failed:", e)
                            try:
                                loader = UnstructuredPDFLoader(file_path)
                                docs = loader.load()
                            except Exception as e:
                                print("UnstructuredPDF failed:", e)
                                continue

                    # ✅ TXT (encoding fix)
                    elif file_path.endswith(".txt"):
                        try:
                            loader = TextLoader(file_path, encoding='utf-8')
                            docs = loader.load()
                        except Exception as e:
                            print("UTF-8 failed:", e)
                            loader = TextLoader(file_path, encoding='latin-1')
                            docs = loader.load()

                    # ✅ Excel
                    elif file_path.endswith((".xls", ".xlsx")):
                        loader = UnstructuredExcelLoader(file_path, mode="single")
                        docs = loader.load()

                    else:
                        print("Unsupported file:", file_path)
                        continue

                    print("Loaded docs:", len(docs))
                    all_documents.extend(docs)

                except Exception as e:
                    print("File failed:", file_path, e)
                    continue

            # ✅ Safety check
            if not all_documents:
                return Response({
                    "error": "No valid content found in files"
                }, status=400)

            # 🔹 Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_documents)

            # 🔹 Vector DB
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text"),
                collection_name="local_rag"
            )

            # 🔹 LLM
            llm = ChatOllama(model="phi3")

            # 🔹 Retriever
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="Generate 2 different rephrased versions of the question: {question}"
            )

            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )

            # 🔹 Prompt
            template = """Answer ONLY using this context:
            {context}

            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            # 🔹 Chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 🔹 Response
            response = chain.invoke(question)

            # cleanup
            vector_db.delete_collection()

            stop = timeit.default_timer()

            return Response({
                'response': response,
                'time': stop - start
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)