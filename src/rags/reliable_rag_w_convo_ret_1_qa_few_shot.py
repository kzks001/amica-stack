import os
import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Semantic chunker import
from langchain_experimental.text_splitter import SemanticChunker

# Constants
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEARCH_TYPE = "mmr"
DEFAULT_RETRIEVAL_K = 5
DEFAULT_FETCH_K = 10
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_MAX_HISTORY_TURNS = 3
DEFAULT_CHUNK_SIZE = 1300
DEFAULT_CHUNK_OVERLAP = int(0.2 * (DEFAULT_CHUNK_SIZE))
DEFAULT_NUM_EXAMPLES = 2
DEFAULT_PERSIST_DIRECTORY = (
    f"data/chroma_db_simple_{DEFAULT_CHUNK_SIZE}_{DEFAULT_CHUNK_OVERLAP}"
)


class HighlightDocuments(BaseModel):
    id: List[str] = Field(
        ..., description="List of id of docs used to answers the question"
    )
    title: List[str] = Field(
        ..., description="List of titles used to answers the question"
    )
    source: List[str] = Field(
        ..., description="List of sources used to answers the question"
    )
    segment: List[str] = Field(
        ...,
        description="List of direct segements from used documents that answers the question",
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class DocumentLoader:
    def load_pdf_documents(self, pdf_dir: str = "data/pdfs") -> List[Document]:
        logger.info(f"Loading PDF documents from {pdf_dir}...")
        docs = []
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_dir, filename)
                logger.debug(f"Loading {file_path}")
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
        return docs


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model_name: str = "text-embedding-3-small",
        breakpoint_threshold_type: str = "standard_deviation",
        breakpoint_threshold_amount: Optional[float] = None,
        use_semantic_chunking: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.use_semantic_chunking = use_semantic_chunking

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if self.use_semantic_chunking:
            logger.info(
                f"Splitting documents using semantic chunking with {self.breakpoint_threshold_type} method..."
            )
            text_splitter = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            )
            chunks = text_splitter.split_documents(documents)
        else:
            logger.info("Splitting documents using character-based chunking...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks


class VectorStoreBuilder:
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        collection_name: str = "simple_rag",
        persist_directory: str = "data/chroma_db_simple",
        batch_size: int = 100,  # Default batch size for processing
    ):
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        model_suffix = embedding_model_name.replace("-", "_").replace(".", "_")
        self.collection_name = f"{collection_name}_{model_suffix}"
        self.vectorstore = None
        self.persist_directory = f"{persist_directory}_{model_suffix}"
        self.batch_size = batch_size

    def _create_batches(self, documents: List[Document]) -> List[List[Document]]:
        """Split documents into batches to avoid token limit issues."""
        batches = []
        current_batch = []
        current_batch_size = 0

        for doc in documents:
            # Rough estimate: 4 chars per token
            doc_tokens = len(doc.page_content) // 4
            if current_batch_size + doc_tokens > 250000:  # Leave some buffer
                batches.append(current_batch)
                current_batch = [doc]
                current_batch_size = doc_tokens
            else:
                current_batch.append(doc)
                current_batch_size += doc_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def build_vectorstore(self, documents: List[Document]) -> Chroma:
        logger.info(f"Building vectorstore with {len(documents)} documents...")

        # Create batches
        batches = self._create_batches(documents)
        logger.info(f"Split documents into {len(batches)} batches")

        # Process first batch to create the collection
        first_batch = batches[0]
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        # Process remaining batches
        for i, batch in enumerate(batches[1:], 1):
            logger.info(f"Processing batch {i + 1}/{len(batches)}")
            self.vectorstore.add_documents(batch)

        self.vectorstore.persist()
        logger.info(f"Vector store saved to {self.persist_directory}")
        return self.vectorstore

    def load_existing_vectorstore(self) -> Chroma:
        logger.info(f"Loading existing vector store from {self.persist_directory}...")
        if not os.path.exists(self.persist_directory):
            logger.error(f"Vector store directory {self.persist_directory} not found")
            raise FileNotFoundError(
                f"Vector store directory {self.persist_directory} not found"
            )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        logger.info("Vector store loaded successfully")
        return self.vectorstore

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        fetch_k: int = None,
        lambda_mult: float = 0.5,
    ):
        if self.vectorstore is None:
            logger.error("Vector store not initialized.")
            raise ValueError(
                "Vector store not initialized. Call build_vectorstore or load_existing_vectorstore first."
            )
        search_kwargs = {"k": k}
        if search_type == "mmr":
            if fetch_k is None:
                fetch_k = max(k * 3, 20)
            search_kwargs.update(
                {
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult,
                }
            )
            logger.info(
                f"Creating MMR retriever with k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}"
            )
        else:
            logger.info(f"Creating similarity retriever with k={k}")
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )


class QueryContextualizer:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_history_turns: int = 3,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.max_history_turns = max_history_turns
        self._setup_contextualizer()

    def _setup_contextualizer(self):
        system_prompt = """You are an AI assistant specializing in reformulating user questions based on conversation context. Your task is to take the user's latest question or statement and transform it into a standalone question that incorporates relevant information from the chat history.

**Critical Instructions:**
- ALWAYS respond with a question that ends with a question mark '?'
- NEVER answer the question or provide any additional information
- If the user's input is already a clear, standalone question, return it unchanged
- If the user's input is not phrased as a question, reformulate it into a question that captures the same intent

**Guidelines for Reformulation:**
- Replace pronouns (it, they, this, that) with specific referents from the chat history
- Include key details such as product names, policy features, or previously mentioned specifics
- Preserve the original meaning and intent of the user's query
- Ensure the reformulated question can be understood without referring back to the chat history

**Final Check:**
Before submitting your response, ensure that:
- It is phrased as a question
- It includes all necessary context from the chat history
- It does not answer the question or provide extra information

Your response should be solely the reformulated question."""
        self.contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self.contextualizer_chain = (
            self.contextualize_prompt | self.llm | StrOutputParser()
        )

    def contextualize_query(self, question: str, chat_history: List) -> str:
        if not chat_history:
            return question
        limited_chat_history = (
            chat_history[-self.max_history_turns * 2 :]
            if len(chat_history) > self.max_history_turns * 2
            else chat_history
        )
        contextualized_question = self.contextualizer_chain.invoke(
            {"question": question, "chat_history": limited_chat_history}
        )
        logger.debug(f"Original question: {question}")
        logger.debug(f"Contextualized question: {contextualized_question}")
        return contextualized_question

    async def contextualize_query_async(self, question: str, chat_history: List) -> str:
        """Async version of contextualize_query for concurrent processing."""
        if not chat_history:
            return question
        limited_chat_history = (
            chat_history[-self.max_history_turns * 2 :]
            if len(chat_history) > self.max_history_turns * 2
            else chat_history
        )
        contextualized_question = await self.contextualizer_chain.ainvoke(
            {"question": question, "chat_history": limited_chat_history}
        )
        logger.debug(f"Original question: {question}")
        logger.debug(f"Contextualized question: {contextualized_question}")
        return contextualized_question


class QueryAtomizer:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._setup_atomizer()

    def _setup_atomizer(self):
        system_prompt = """You are an expert at breaking down complex questions into simpler, atomic sub-questions. Your task is to decompose the given question into distinct, standalone sub-questions that each capture a specific aspect of the original intent.

Key guidelines:
- Each sub-question should be self-contained and answerable independently
- Sub-questions should not overlap in scope
- Maintain the original meaning and intent
- Focus on insurance-specific aspects when relevant
- Return the sub-questions as a numbered list
- If the question is already atomic, return it as is

Use case specific instructions:
- If the question is about LOG, return the sub-questions for eLOG as well.

Example:
Input: "What are the coverage limits and exclusions for cancer treatment under this policy?"
Output:
1. What are the coverage limits for cancer treatment under this policy?
2. What are the exclusions for cancer treatment under this policy?"""
        self.atomize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        self.atomizer_chain = self.atomize_prompt | self.llm | StrOutputParser()

    def atomize_query(self, question: str) -> List[str]:
        logger.info(f"Atomizing query: {question}")
        atomized_queries = self.atomizer_chain.invoke({"question": question})
        queries = []
        for line in atomized_queries.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                query = line.split(". ", 1)[1].strip()
                queries.append(query)
            elif line and not line[0].isdigit():
                queries.append(line)
        if not queries:
            queries = [question]
        logger.debug(f"Atomized queries: {queries}")
        return queries

    async def atomize_query_async(self, question: str) -> List[str]:
        """Async version of atomize_query for concurrent processing."""
        logger.info(f"Atomizing query: {question}")
        atomized_queries = await self.atomizer_chain.ainvoke({"question": question})
        queries = []
        for line in atomized_queries.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                query = line.split(". ", 1)[1].strip()
                queries.append(query)
            elif line and not line[0].isdigit():
                queries.append(line)
        if not queries:
            queries = [question]
        logger.debug(f"Atomized queries: {queries}")
        return queries


class ResponseGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_history_turns: int = 3,
        use_few_shot: bool = True,
        num_examples: int = 2,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
        self.max_history_turns = max_history_turns
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples

        # Initialize example selector with insurance domain examples
        self.initialize_examples()
        self._setup_generator()

    def initialize_examples(self):
        """Initialize insurance domain examples for few-shot prompting."""
        examples = [
            # 1.1
            {
                "question": "My client is on Singlife Shield Plan 1 with Singlife Health Plus Private Lite Plan He is unable to pay for his premiums and can he request for extension of the payment deadline?",
                "answer": "No, the payment deadline will not be extended. Premiums are required to be paid within 60 days from the renewal date. If payment is not made within 60 days from the renewal date, the policy will lapse in accordance with its terms and conditions. Additionally, if the premium is not paid on time, we may deduct the premium from the Medisave account as permitted by regulations.",
            },
            # 1.2
            {
                "question": "What if my client misses a payment past the grace period?",
                "answer": "Your client has the option to reinstate the policy within 30 days following the lapse date, provided the lapse was due to non-payment of premium, and the reinstatement is contingent upon the following conditions: First, the life assured must be 75 years old or younger at the time of reinstatement. Second, all outstanding premiums must be paid prior to the reinstatement of the policy. Third, satisfactory proof indicating that each life assured is insurable must be arranged and funded by you. Furthermore, the reinstatement may be subject to exclusion if there are changes in the medical condition of the life assured.",
            },
            # 1.3
            # {
            #     "question": "Also, he is turning 61 on his  next birthday. Will this result in an increase in my premium? If so, by how much?",
            #     "answer": "Yes, your premiums will increase by $376 when your client turns 61 on his next birthday. For the Singlife Shield Plan 1 combined with the Singlife Health Plus Private Lite Plan 1 that is renewed on or after 1 September 2024, the estimated premium for policyholders who are 60 years old on their next birthday is $2,571. The components of this amount are as follows: the premium for the Singlife Shield Plan 1 is $1,892, and when adding the premium for the Singlife Health Plus Private Lite Plan 1, which is $679, it totals $2,571. \n\nOn the other hand, for the same plans renewed on or after 1 September 2024, the estimated premium for policyholders who are 61 years old on their next birthday is $2,947.38. This total can be broken down into the premium for the Singlife Shield Plan 1, which is $2,040.00, and the premium for the Singlife Health Plus Private Lite Plan 1, which amounts to $907.00.",
            # },
            # 1.4
            {
                "question": "Does Singlife Health Plus cover critical illness coverage?",
                "answer": "Singlife Health Plus provides coverage for critical illness up to $10,000, provided that the person insured has celebrated their first birthday and is not older than 65 years.",
            },
            # 1.5
            # {
            #     "question": "Is there an option for him to upgrade or downgrade  coverage for my Shield and Health Plus policies?",
            #     "answer": "Yes, it is feasible to modify the plan options, whether by upgrading or downgrading the chosen option for the life assured. \n\nWhen upgrading the life assured's option, it is necessary to provide satisfactory proof of insurability for each individual covered by the policy, and any associated costs must be paid by the policyholder. Additionally, any claims that arise from a pre-existing condition after the upgrade will be evaluated according to the terms and conditions of the previous option.\n\nOn the other hand, when downgrading the life assured's option, there is no obligation to declare any medical conditions. \n\nAfter the request to change the life assured's option has been approved, we will notify you in writing of the effective date for the new option.",
            # },
            # 1.6
            # {
            #     "question": "My client is keen is interested in adding comprehensive cancer coverage. Can we add Cancer Cover Plus as a rider?",
            #     "answer": "Singlife Cancer Cover Plus II cannot be added as a rider because it is a standalone plan. Your client is able to purchase Singlife Cancer Cover Plus II independently, without the need for an Integrated Shield Plan.",
            # },
            # 1.7
            {
                "question": "Does Cancer Cover Plus cover the removal of benign cysts or growths?",
                "answer": "Yes, it includes the removal of certain benign cysts or growths. For benign growths, the following can be accepted as long as no treatment or referral to a specialist is required: adenomyosis or endometriosis, cervical cysts, chalazia, dermoid cysts, keratinous cysts, sebaceous cysts, nabothian cysts, and spinal synovial cysts. \n\nFurthermore, simple breast cysts and other types of growths that have been removed without recurrence and do not require follow-up can also be accepted. These include simple breast cysts, congenital brain cysts such as arachnoid or colloid cysts, endometrial polyps, gallbladder polyps, hemangiomas, lipomas, ovarian cysts, pilonidal cysts, rhabdomyomas, and uterine fibroids. \n\nHowever, it cannot be claimed if there was an exclusion related to breast disorders.",
            },
            # 1.8
            {
                "question": "Does Cancer Cover Plus provide coverage for overseas treatment?",
                "answer": "Yes, the Singlife Cancer Cover Plus II policy will cover the medical expenses that the life assured incurs at a hospital overseas if the life assured needs cancer treatments that are included in the policy, given that two specific conditions are satisfied. The first condition is that the life assured must not have been outside of Singapore for more than 183 consecutive days within a 12-month period prior to their admission date. The second condition is that we must have pre-approved these expenses and issued a certificate of pre-authorization.\n\nHowever, the policy excludes coverage for several items. It will not cover experimental or pioneering medical or surgical techniques, as well as medical devices, which include treatments that are investigational or research-based and not approved by the relevant regulatory bodies. Additionally, it will not cover clinical trials for medicinal products, regardless of whether these trials have received clinical trial certificates from the appropriate regulatory bodies. Moreover, medical devices, drugs, therapeutic products, and cell, tissue, and gene therapy products that are not approved or are not used according to the indications specified by the relevant regulatory bodies will also not be covered.",
            },
            # 1.9
            {
                "question": "Can client use his Medisave account to pay the premium for the Cancer Cover Plus plan?",
                "answer": "The client is not permitted to utilize MediSave to cover the premiums for this policy since Singlife Cancer Cover Plus II is not approved for MediSave. Instead, the client can employ the following payment methods:\n\nFor the first premium payment during the application process, the available options include credit card payments via Visa or Mastercard, eGIRO for DBS or POSB bank accounts only, and self-initiated payments such as AXS or Internet Banking Funds Transfer.\n\nFor subsequent premium payments, the options consist of Interbank GIRO and self-initiated payments like AXS or Internet Banking Funds Transfer.",
            },
            # 1.10
            # {
            #     "question": "Does he need to file separate claims for my Cancer Cover Plus and Shield plan?",
            #     "answer": "Your client does not need to submit a separate claim for Cancer Cover Plus II if they possess Singlife's Integrated Shield Plan (IP). Instead, the Cancer Cover Plus II claim will be processed in conjunction with any e-filed Singlife Shield or Singlife Health Plus claim submitted to Singlife. The only exception to this is for treatment received overseas, for which a separate submission will be necessary.",
            # },
            # 1.11
            # {
            #     "question": "Is there a way to assess to check possible underwriting results before making any submissions?",
            #     "answer": "Yes, you are able to obtain preliminary underwriting advice for complex cases through Singlife Online (SOL). To do this, please log in to SOL, navigate to the "Services" tab, and choose "Preliminary Underwriting" to submit your request. Please be aware that requests sent via email will not be accepted.\n\nTypically, our underwriters will respond within three business days.\n\nOnce your preliminary underwriting request has been reviewed, you will receive an email notification. To view the underwriting decision, you should log in to SOL, select the "Services" tab, and click on "Preliminary Underwriting." Then, you can access "My Enquiries," which is located at the top right corner of the Preliminary Underwriting page.",
            # },
            # 1.12
            # {
            #     "question": "If my client wishes to push through with the application for Cancer Cover Plus, what are the documents required?",
            #     "answer": "Advisers are strongly encouraged to utilize the complete digital submission method, which involves the use of EzSub, MyInfo, and/or eFNA/eFP. \n\nThe submission must include the following documents: \n\n1. The Application Form for Singlife Cancer Cover Plus.\n2. The Product Summary.\n3. The relevant sections of the Life Insurance Advisory Form, commonly known as the Fact Find form, which include Section 11 - Declaration by Representative, Section 12 - Acknowledgement by Client, and Section 13 - Supervisor's Review. \n4. The Application for Interbank GIRO.\n5. A photocopy of the ID (Identity Document) or Passport of the proposer (the assured) and any dependent(s). Alternatively, MyInfo may be used to expedite the processing time. \n6. Citizenship Papers for children if their Birth Certificates indicate that they were not citizens of Singapore at the time of their birth. \n7. Proof of residential address; please consult the list of acceptable documents for this purpose. \n8. The Non-Face to Face Supplementary Form, if applicable. \n9. Citizenship Papers for children if their Birth Certificates indicate that they were not citizens of Singapore at the time of their birth. \n\nIt is important to note that for the Affinity Channel, a complete set of Fact Find documents must be submitted.",
            # },
            # 1.13
            # {
            #     "question": "I don't think I can find in EzSub a way to monitor the status of the application. Is there any other way to monitor the status of the application?",
            #     "answer": "You are able to monitor the application's status through the Singlife Online (SOL) Dashboard by selecting the option labeled 'Pending Underwriting/Further Requirements.' Furthermore, the New Business team will send any correspondence regarding additional requirements to the corporate email address of the Financial Adviser Representative.",
            # },
            # 1.14
            # {
            #     "question": " If a medical report is required will Singlife bear the cost?",
            #     "answer": "Singlife does not cover the costs associated with obtaining a medical report if it is required. It is the responsibility of the applicant to provide all necessary medical evidence, as Singlife does not organize medical examinations or secure medical reports on behalf of the applicant.",
            # },
            # 2.1
            # {
            #     "question": "Is Singlife Shield Plan 1 with Health Plus Private Prime, the highest hospital plan offered by Singlife?",
            #     "answer": "Correct.",
            # },
            # 2.2
            {
                "question": "For Singlife Shield Plan 1 with Health Plus Private Prime, please explain what is the difference between choosing a panel specialist and choosing a non-panel specialist.",
                "answer": "The information provided is accurate. The annual deductible is set at $0, which is a reduction from the previous amounts of $1,000 for inpatient care and $500 for day surgery. Additionally, the co-insurance rate is 5%, but it is limited to a maximum of $3,000 per policy year, rather than the previous cap of $12,750 per policy year.",
            },
            # 2.3
            #     "question": "Where can I find a list of your panel specialists to choose from?",
            #     "answer": "You can request an appointment by visiting singlife.com/medicalspecialists, calling 1800 600 0066, or using our Singlife App or MySinglife Portal. Please note that the appointment request service is available only for the initial visit to the panel specialist. For any follow-up visits, you will need to schedule the appointment directly with the clinic.",
            # },
            # 2.4
            {
                "question": "For Singlife Shield Plan 1 with Health Plus Private Prime, please explain what do I need to do to enjoy cashless admission to hospitals.",
                "answer": "The admission deposit can be waived if the eLOG is applied, the bill size exceeds the deductible, and the reason for hospitalization is not included in the general exclusions. There are different waiver amounts based on the type of hospital: for a public hospital, the waiver can be up to S$80,000; for a panel specialist in a private hospital, it can be up to S$50,000; and for a non-panel specialist in a private hospital, it can be up to S$15,000. \n\nWhen you are admitted or on the day of your surgery, the hospital staff will verify your eligibility for the waiver of the admission deposit through the eLOG system. The eLOG system allows for the waiver of the admission deposit required by participating hospitals if the estimated medical bill exceeds the plan's deductible. However, it is important to note that the eLOG is subject to the hospital's acceptance and does not guarantee a waiver of the deposit. At the time of discharge, the hospital may still require the patient to pay the full hospital bill, even if an eLOG has been issued. While this service is provided to facilitate the admission process by eliminating the need for upfront cash up to the approved amount by the eLOG system, Singlife reserves the right to review each claim submitted after discharge. If the claim is deemed payable, Singlife will cover the eligible claim amount; if not, Singlife or the hospital will request payment for any amount not covered by the policy.\n\nThe list of hospitals and clinics that participate in this program will be updated periodically, and you can find the latest information on our website at singlife.com/en/medical-insurance/shield/log. \n\nThe eLOG service is governed by several key terms and conditions: the hospital may still require the patient to pay the full bill even if an eLOG has been issued; an eLOG will not be issued if the estimated medical bill is below the annual deductible or if the medical condition to be treated is listed as an exclusion in the policy document; the annual deductible and/or co-insurance will not be included in the eLOG unless the patient is also covered under Singlife Health Plus; the eLOG is not a benefit of the policy and is not included in the Singlife Shield policy document; and the issuance of an eLOG is subject to Singlife's review and discretion, meaning it does not imply approval or acceptance of any claims made under the Singlife Shield or Singlife Health Plus policy. Singlife will evaluate the claim upon receiving the hospital bill.\n\nHospital staff can generate the Singlife eLOG instantly by accessing the eLOG system. It is important to note that we do not provide eLOGs for non-participating hospitals, which will operate solely on a reimbursement basis. However, the hospital can assist in electronically filing the claim for you.",
            },
            # 2.5
            {
                "question": "Please explain what are the hospital bill limits covered by Singlife Shield Plan 1 with Health Plus Private Prime.",
                "answer": "The annual deductible is structured such that if the treatment is from a panel provider, it is not payable. However, if the treatment is from a non-panel provider, the deductible is payable at $1,000 for inpatient care or $500 for day surgery. Regarding co-insurance, if the treatment is from a panel provider, it is payable at 5%, with a cap of $3,000 per policy year. In contrast, if the treatment is from a non-panel provider, it is also payable at 5%, but with a higher cap of $12,750 per policy year.\n\nFor outpatient cancer drugs that are not part of the Common Drug List (Non-CDL), the benefit is capped at S$30,000 per policy year, subject to co-insurance. Ambulance fees or transport to the hospital are covered at S$80 per injury or illness. Additionally, accommodation charges for a parent or guardian of a child life assured are covered at S$80 per day, up to a maximum of 10 days.\n\nThe ward type covered is any standard ward in a private hospital. Pre-hospital treatment is covered as charged for up to 180 days before admission if provided by a panel specialist in a private hospital with a certificate of pre-authorization, or in a public hospital, community hospital, MOH-approved Inpatient Hospice Palliative Care Service (IHPCS) provider, or in the Accident & Emergency department. Alternatively, if the treatment is from an extended panel or a non-panel specialist in a private hospital, or a panel specialist in a private hospital without a certificate of pre-authorization, it is covered as charged for up to 90 days before admission.\n\nPost-hospital treatment is covered as charged for up to 365 days after discharge if provided by a panel specialist in a private hospital with a certificate of pre-authorization, or in a public hospital, community hospital, MOH-approved Inpatient Hospice Palliative Care Service (IHPCS) provider, or in the Accident & Emergency department. If the treatment is from an extended panel, a non-panel specialist in a private hospital, or a panel specialist in a private hospital without a certificate of pre-authorization, it is covered as charged for up to 180 days after discharge.\n\nFor outpatient cancer drug treatment that is on the Common Drug List (CDL), the limit is five times the MediShield Life claim limit per month. Outpatient cancer drug services are capped at five times the MediShield Life claim limit per policy year. The coverage for inpatient and outpatient Proton Beam Therapy treatment is S$70,000 per policy year, while inpatient and outpatient Cell, Tissue, and Gene Therapy is capped at S$150,000 per policy year. The policy year limit is set at S$2,000,000 for panel providers and S$1,000,000 for non-panel providers. For more detailed information, including limits of liability, clients are advised to refer to the policy contract, as the complete details are too extensive to provide in a chatbot response.",
            },
            # 2.6
            {
                "question": "For Singlife Shield Plan 1 with Health Plus Private Prime, is there ward downgrade benefit?",
                "answer": "This does not apply to Private Prime.",
            },
            # 2.7
            # {
            #     "question": "Can you explain on CDL vs non CDL? I am not sure what is covered or not.",
            #     "answer": 'The term "Cancer Drug List" or "CDL" refers to the compilation of cancer drug treatments that have been clinically validated and are considered more cost-effective, which can be found on the Ministry of Health (MOH) website at https://go.gov.sg/moh-cancerdruglist. It is important to note that outpatient cancer drug treatments can only be claimed under your policy if they are utilized in accordance with the clinical indications outlined in the CDL, unless your policy specifies otherwise. The MOH may periodically update the CDL.\n\nOn the other hand, "Non Cancer Drug List" or "Non-CDL" treatments refer to cancer drug treatments that are not included in the Cancer Drug List and are categorized as Non-CDL treatments according to the Non-CDL Classification Framework established by the Life Insurance Association of Singapore, which can be accessed at https://www.lia.org.sg/media/3553/non-cdl-classification-framework.pdf.\n\nWe will provide coverage for outpatient cancer drug treatments classified under Classes A to E of the Non-CDL Classification Framework, as detailed in the aforementioned link, up to the limits specified in the benefits schedule and subject to co-insurance. However, treatments classified as Class F are not covered. It is also important to note that outpatient cancer drug treatments that fall under the Non-CDL category are not subject to the maximum co-insurance limits stated in the benefits schedule.\n\nYour current coverage includes the following: for outpatient cancer drug treatment on the CDL, you have a limit of five times the MediShield Life (MSHL) limit per month, and for outpatient cancer drugs under the Non-CDL category, you have a benefit of S$30,000 per policy year. Starting from September 1, 2024, Singlife Health Plus policies will provide additional coverage for cancer drug treatments listed on the Cancer Drug List (CDL), enhancing the existing five times MSHL limit available under Singlife Shield plans, and will also offer increased coverage for cancer drug treatments that are not listed on the CDL. Upon renewing your policy next year, your coverage will be adjusted to the following: for outpatient cancer drug treatment on the CDL, the limit will increase to twenty times the MSHL limit per month, and for outpatient cancer drugs under the Non-CDL category, the benefit will rise to S$180,000 per policy year, with a maximum of S$15,000 per month.',
            # },
            # 2.8
            # {
            #     "question": "The monthly limits may not be sufficient! I also heard that you all have a cancer plan, what's the name and hows the coverage like? And is my wife still eligible to sign up now? Or her condition will be excluded? ",
            #     "answer": "The product is named Singlife Cancer Cover Plus II. This plan is a non-participating, yearly renewable medical reimbursement option that offers protection against outpatient cancer drug treatments and services, which include drugs listed on both the Cancer Drug List (CDL) and the non-Cancer Drug List (non-CDL). Additionally, it covers selected cancer treatments such as Proton Beam Therapy and Cell Tissue and Gene Therapy. Policyholders can enjoy greater peace of mind knowing that significant medical expenses will be reimbursed as charged. This plan is particularly suitable for those utilizing private care services that require higher coverage and offers both local and overseas protection. It is important to note that this plan does not accumulate any cash value. Furthermore, the premium rates are not guaranteed; they may be reviewed and adjusted periodically, with at least 30 days' written notice provided to policyholders before any changes take effect upon renewal. Additionally, this policy is not approved for MediSave usage, meaning that premiums cannot be paid using MediSave funds.\n\nThe Benefits Schedule is detailed in Appendix B, which includes a table with footnotes.\n\nTo qualify for Singlife Cancer Cover Plus II, the life assured must be a Singapore citizen or a Singapore permanent resident and must be 75 years old or younger at the next birthday on the cover start date. Dependants of the life assured are also eligible for coverage under this plan, provided they are Singapore citizens or Singapore permanent residents. A newborn can be covered 15 days after birth or after being discharged from the hospital, whichever occurs later.\n\nA pre-existing condition refers to any condition or illness that was present or evident, or for which the life assured was suffering, prior to the policy issue date, cover start date, or last reinstatement date, whichever is the most recent, unless the condition or illness was declared and accepted by the insurer. All pre-existing conditions are excluded from coverage unless they have been disclosed and accepted in writing by the insurer.",
            # },
            # 2.9
            # {
            #     "question": "Whats the premium amount like for 45 year old?",
            #     "answer": "The annual premium for a 46-year-old individual is S$280, while the monthly premium is S$23.88. It is important to note that premiums are not guaranteed, and Singlife reserves the right to review and adjust the premium rates periodically, providing at least 30 days' notice before any changes take effect at the next premium due date. Although premium rates are subject to regular review on a portfolio basis, individual insured parties will not face penalties for poor claims or health issues. The premium rates include Goods & Services Tax (GST) at the current rate and are determined based on the age of the insured individual as of their next birthday. Additionally, there is no difference in premium rates between males and females, and the rates vary according to the life assured's age band, which is based on their age next birthday at the time of renewal.",
            # },
            # 2.10
            # {
            #     "question": "Oh ya, breast cancer is considered a critical illness right? Is there payout and how to claim?",
            #     "answer": "The critical illness benefit is set at S$10,000 for the lifetime of the insured individual, applicable only if the insured has surpassed their first birthday and is not older than 65 years old at their next birthday. We will disburse the critical illness benefit provided that the insured individual is diagnosed with any critical illness and remains alive after the designated survival period. In cases where the diagnosed critical illness includes Major Cancer, Coronary Artery Bypass Surgery, Angioplasty and Other Invasive Treatments for Coronary Artery, Other Serious Coronary Artery Disease, or Heart Attack of Specified Severity, the benefit will only be payable if the diagnosis occurs after a waiting period of 90 days from either the start date of the coverage or the last reinstatement date, whichever is later.\n\nTo claim the critical illness benefit, you must provide us with written notice within 30 days following the occurrence of an accident or the diagnosis of the insured's critical illness. We will accept any written notice submitted on behalf of the insured that includes sufficient details for us to identify the insured individual. If the notice is not provided within the specified timeframe, you may still file a claim if you can demonstrate that it was not reasonably possible to give such notice and that you notified us as soon as it became reasonably feasible. Additionally, to facilitate the processing of your claim, you must provide us with any or all of the following at your own expense, if we request them: certificates, medical reports, information, and evidence in the format and nature we require; evidence to confirm the ongoing health condition of the insured; and the insured must be examined by our approved doctor. If the insured resides outside of Singapore, we may require them to travel to Singapore for the examination. Furthermore, you must provide proof of the insured's date of birth. If the date of birth or age provided to us is incorrect, we will only pay the amount that corresponds to what we would have paid had the correct date of birth or age been provided.\n\nIt is essential that all the requirements outlined in the previous clause are fulfilled. If, based on medical facts and a balance of probabilities, we find it appropriate to deny the claim in accordance with clause 7.9 of your Policy Contract (Pre-existing conditions), the responsibility lies with the insured to present any evidence we may reasonably require to prove otherwise, allowing us to reconsider the claim.",
            # },
            # 3.1
            # {
            #     "question": "Hi, I am looking at your cancer cover plus 2 plan on your website but cannot find anything about payment methods. Can I pay the premiums with credit card?",
            #     "answer": "The first premium is required, while subsequent premiums are not. You have the option to set up eGIRO for both the first and subsequent payments. Alternatively, you can initiate payments yourself through AXS or by bank transfer.",
            # },
            # 3.2
            # {
            #     "question": "For Singlife Shield, what is the easiest and quickest way to setup GIRO?",
            #     "answer": "DBS account holders have the opportunity to easily and quickly establish a recurring payment for their regular premium renewal policies in Singapore dollars by enrolling in eGIRO through MySinglife. \n\nTo do this, first log in to your MySinglife account. Next, navigate to the "Services" section and select "Policy Servicing." From there, choose the option to "Set up eGIRO for Premium Payment" and select your desired Policy Number. After clicking "Next," you will need to update your bank account details and select the bank you wish to use for enrollment. Once you click on "Set up eGIRO," you will be redirected to the internet banking website of your chosen bank. Please log in using your banking User ID and password. Finally, enter the one-time password (OTP) or the login authentication notification sent by your bank to complete the transaction.\n\nIt is important to note that the GIRO application approval process typically takes only a few minutes. For additional information, please visit our website FAQ at https://singlife.com/en/policy-servicing#premium-payments.",
            # },
            # 3.3
            # {
            #     "question": "For Singlife Shield, when are the GIRO deduction dates?",
            #     "answer": "The first deduction will occur on the 7th of the month, while the second deduction will take place on the 20th of the month. If either of these deduction dates falls on a non-working day, the deduction will be rescheduled to the next working day.",
            # },
            # 3.4
            # {
            #     "question": "For my Singlife Shield plan that is on GIRO payment, what happens if all GIRO deduction attempts fail? ",
            #     "answer": "You are granted a grace period of 60 days from the renewal date to pay your premium. During this grace period, your policy remains active. However, you must settle any outstanding premiums or amounts owed to us before we can process any claims under your policy. If the premium is not paid by the final day of the grace period, your policy will terminate on the renewal date.\n\nOnce your policy ends, you will have no further claims or rights against us, even if your claim is related to a covered condition that occurred prior to the termination of your policy. The termination of your policy does not impact the coverage of the life assured under MediShield Life, which will continue as long as the life assured remains eligible according to the relevant act and regulations.\n\nIf your policy is terminated due to non-payment of the premium, you have the option to apply for reinstatement within 30 days from the date you receive the termination notice, provided you meet the following conditions: the life assured must be 75 years old or younger on the reinstatement date, you must pay all premiums owed before we will reinstate your policy, and you must provide satisfactory proof of insurability for each life assured at your own expense. If we agree to reinstate your policy, we will send you a notice of reinstatement. Should there be any changes in the medical or physical condition of the life assured, we may impose exclusions from the reinstatement date. To clarify, if we accept any premium after your policy has ended, it does not imply that we will waive our rights under your policy or assume any liability regarding claims. We will not cover any treatment provided to the life assured after the policy termination date and within 30 days of the reinstatement date, except for inpatient treatment for injuries resulting from an accident that occurred after the reinstatement date.",
            # },
            # 3.5
            # {
            #     "question": "Singlife Shield plan, please explain on reinstatement requirements, procedures, and any other useful info, if my policy lapse from non-payment.",
            #     "answer": "To reinstate the effective coverage of your policy, which is referred to as the Reinstatement of the Policy, you need to declare your health status by filling out the health declaration form. For health policies, this is known as the Policy Servicing Health Declaration for Health Products. \n\nYou must also pay any outstanding premiums along with any accumulated interest, if applicable, and any existing indebtedness. The reinstatement process is subject to underwriting approval, and additional requirements, such as medical reports, may be requested during the review. If your health status has worsened, you might be required to pay a higher premium based on the underwriter's assessment.\n\nWe suggest using interbank GIRO for your premium payments, as this will automate the payment process for you. To initiate this, please submit a GIRO application. \n\nFor further details, you can visit our website's FAQ section at: https://singlife.com/en/policy-servicing#paying-for-your-policy&reviving-a-lapsed-policy.",
            # },
        ]

        # Create example selector
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL),
            Chroma,
            k=self.num_examples,
            input_keys=["question"],
        )

    def _setup_generator(self):
        system = """You are an AI assistant who is helpful, respectful, and honest. For each user query, you will be provided with context information that may consist of a combination of text, tables, and/or visual data. Your job is to use the provided context to accurately answer the user question. 
        
        Key guidelines for handling chat history and context: 
        - Pay careful attention to the chat history to understand the full context of the conversation.
        - If the current question refers to previous messages (using terms like "it", "this policy", "that coverage"), use the chat history to resolve these references.
        - Maintain a conversational and natural tone throughout your responses.
        - If a question seems ambiguous or incomplete, look at the chat history for clarifying information.
        - The user may ask follow-up questions that build on previous questions and answers.
        
        Guidelines for using retrieved documents:
        - Use the context provided in the retrieved documents.
        - Your answers should be strictly based on the information given in the context. 
        - Do not make up answers.
        
        Insurance-specific guidelines:
        - For What-If scenarios (e.g., what happens if the client misses a payment or what happens if policy lapses), explain what will happen and what the client can do to reinstate and reinstate conditions. 
        - When asked if something is covered or if a rider provides coverage (e.g., Does the rider cover...? or is there coverage for...?), your response should be: Clear and direct answer: Start with a yes or no answer. If the answer is yes, follow up with the coverages, benefits, and payouts. The answer does not need to include according to which document and no need to include any additional or noting information, just provide the answer based on the context. You do not need to provide summary and even if you do, you need to ensure the summary stays within the context that you are given. Ensure length of the answer is within 350 words. 
        
        Key guidelines for response:
        - Always answer the question as directly and succinctly as possible.
        - It is acceptable to respond with very short answers when appropriate (e.g., "Correct").
        - Begin your response with a clear and concise answer to the question.
        - If additional explanation is needed (e.g., in "What-If" scenarios), provide the details immediately after the direct answer.
        - If the the response is a step-by-step guide or how-to instructions, you may exceed the 350-word limit to ensure clarity.

        Examples: 
        - If the user asks, What if my client misses a payment after the grace period?, your answer should describe what happens but also suggest what actions the client can take (e.g., all the conditions and eligibility for reinstating the policy). 
        - If the user asks, Does Cancer Cover Plus provide overseas treatment coverage?, your answer should be yes, but follow up with specific details about the coverage, such as what types of overseas treatments are covered, any geographical restrictions, and any conditions that apply. 
        - If the user asks, Does the rider includes critical illness coverage? if your answer is yes, also follow up with the benefits, critical illness coverage and payouts. 
        - If the user asks about setting up Giro or asking about Giro deductions, always also include eGiro context in the answer.
        """

        if self.use_few_shot:
            # Create an example prompt template
            example_prompt = PromptTemplate(
                input_variables=["question", "answer"],
                template="Question: {question}\n\nAnswer: {answer}",
            )

            # Create the few-shot prompt with examples
            few_shot_prompt = FewShotPromptTemplate(
                example_selector=self.example_selector,
                example_prompt=example_prompt,
                prefix=system
                + "\n\nHere are some examples of insurance-related questions and answers:",
                suffix="\n\nChat History:\n{chat_history}\n\nRetrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>\n\nAnswer:",
                input_variables=["documents", "question", "chat_history"],
            )

            # Chain the few-shot prompt with the LLM
            self.rag_chain = few_shot_prompt | self.llm | StrOutputParser()
        else:
            # Standard prompt without few-shot examples
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder(variable_name="chat_history"),
                    (
                        "human",
                        "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>",
                    ),
                ]
            )

            self.rag_chain = prompt | self.llm | StrOutputParser()

    def format_docs(self, docs: List[Document]) -> str:
        return "\n".join(
            f"<doc{i + 1}>:\nTitle:{doc.metadata.get('title', 'Unknown')}\nSource:{doc.metadata.get('source', 'Unknown')}\nContent:{doc.page_content}\n</doc{i + 1}>\n"
            for i, doc in enumerate(docs)
        )

    def generate_response(
        self, docs: List[Document], question: str, chat_history: Optional[List] = None
    ) -> str:
        logger.info("Generating response...")
        formatted_docs = self.format_docs(docs)
        limited_chat_history = []
        if chat_history:
            limited_chat_history = (
                chat_history[-self.max_history_turns * 2 :]
                if len(chat_history) > self.max_history_turns * 2
                else chat_history
            )

        # Format chat history as a string for the few-shot prompt
        chat_history_str = ""
        if limited_chat_history:
            chat_history_str = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in limited_chat_history]
            )

        return self.rag_chain.invoke(
            {
                "documents": formatted_docs,
                "question": question,
                "chat_history": (
                    chat_history_str
                    if self.use_few_shot
                    else limited_chat_history or []
                ),
            }
        )

    def generate_streaming_response(
        self, docs: List[Document], question: str, chat_history: Optional[List] = None
    ):
        """Generate a streaming response for the given question and documents."""
        logger.info("Generating streaming response...")
        formatted_docs = self.format_docs(docs)
        limited_chat_history = []
        if chat_history:
            limited_chat_history = (
                chat_history[-self.max_history_turns * 2 :]
                if len(chat_history) > self.max_history_turns * 2
                else chat_history
            )

        # Format chat history as a string for the few-shot prompt
        chat_history_str = ""
        if limited_chat_history:
            chat_history_str = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in limited_chat_history]
            )

        # Return the stream generator
        return self.rag_chain.stream(
            {
                "documents": formatted_docs,
                "question": question,
                "chat_history": (
                    chat_history_str
                    if self.use_few_shot
                    else limited_chat_history or []
                ),
            }
        )


class DocumentHighlighter:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_history_turns: int = 3,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=HighlightDocuments)
        self.max_history_turns = max_history_turns
        self._setup_highlighter()

    def _setup_highlighter(self):
        system = """You are an advanced assistant for document search and retrieval in the insurance domain. You are provided with the following:
        1. A question about insurance policies, coverage, or related topics.
        2. A generated answer based on the question.
        3. A set of documents that were referenced in generating the answer.
        4. The chat history of the conversation (if available).

        Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
        generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
        in the provided documents.

        Ensure that:
        - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
        - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
        - (Important) If you didn't use a specific document, don't mention it.
        - Consider the context of the entire conversation when identifying relevant segments.
        - Focus on insurance-specific content that supports claims about policy coverage, exclusions, or benefits.
        - When highlighting segments related to insurance terms, ensure they accurately represent policy conditions described in the answer.
        """
        prompt = PromptTemplate(
            template=(
                system + "\n\n"
                "Chat History: {chat_history}\n\n"
                "Used documents: <docs>{documents}</docs>\n\n"
                "User question: <question>{question}</question>\n\n"
                "Generated answer: <answer>{generation}</answer>\n\n"
                "{format_instructions}"
            ),
            input_variables=["documents", "question", "generation", "chat_history"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.doc_lookup = prompt | self.llm | self.parser

    def highlight_documents(
        self,
        docs: List[Document],
        question: str,
        generation: str,
        chat_history: Optional[List] = None,
    ) -> HighlightDocuments:
        logger.info("Highlighting relevant document segments...")
        formatter = ResponseGenerator()
        formatted_docs = formatter.format_docs(docs)
        formatted_chat_history = ""
        if chat_history and len(chat_history) > 0:
            limited_chat_history = (
                chat_history[-self.max_history_turns * 2 :]
                if len(chat_history) > self.max_history_turns * 2
                else chat_history
            )
            formatted_chat_history = "\n".join(
                [f"{msg.type}: {msg.content}" for msg in limited_chat_history]
            )
        try:
            return self.doc_lookup.invoke(
                {
                    "documents": formatted_docs,
                    "question": question,
                    "generation": generation,
                    "chat_history": formatted_chat_history,
                }
            )
        except Exception as e:
            logger.error(f"Error highlighting documents: {str(e)}")
            fallback_ids = [f"doc{i + 1}" for i in range(len(docs))]
            fallback_titles = [doc.metadata.get("title", "Unknown") for doc in docs]
            fallback_sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            fallback_segments = []
            for doc in docs:
                text = doc.page_content
                first_para = text.split("\n\n")[0] if "\n\n" in text else text[:200]
                fallback_segments.append(first_para)
            return HighlightDocuments(
                id=fallback_ids,
                title=fallback_titles,
                source=fallback_sources,
                segment=fallback_segments,
            )


class DocumentGrader:
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self._setup_grader()

    def _setup_grader(self):
        system = """You are a grader assessing relevance of a retrieved document to a user question. 

        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )
        self.retrieval_grader = grade_prompt | self.structured_llm_grader

    def grade_document(self, document: Document, question: str) -> GradeDocuments:
        """Grade a single document for relevance to the question."""
        return self.retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )

    async def grade_document_async(
        self, document: Document, question: str
    ) -> GradeDocuments:
        """Async version of grade_document for concurrent processing."""
        return await self.retrieval_grader.ainvoke(
            {"question": question, "document": document.page_content}
        )

    def filter_relevant_documents(
        self, documents: List[Document], question: str
    ) -> List[Document]:
        """Filter documents based on relevance to the question."""
        logger.info(f"Grading {len(documents)} documents for relevance...")
        relevant_docs = []

        for doc in documents:
            try:
                grade_result = self.grade_document(doc, question)
                if grade_result.binary_score == "yes":
                    relevant_docs.append(doc)
                    logger.debug(
                        f"Document marked as relevant: {doc.metadata.get('title', 'Unknown')}"
                    )
                else:
                    logger.debug(
                        f"Document marked as irrelevant: {doc.metadata.get('title', 'Unknown')}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error grading document {doc.metadata.get('title', 'Unknown')}: {str(e)}"
                )
                # In case of error, include the document to be safe
                relevant_docs.append(doc)

        logger.info(
            f"Filtered to {len(relevant_docs)} relevant documents out of {len(documents)}"
        )
        return relevant_docs

    async def filter_relevant_documents_async(
        self, documents: List[Document], question: str
    ) -> List[Document]:
        """Async version of filter_relevant_documents for concurrent processing."""
        logger.info(f"Grading {len(documents)} documents for relevance (async)...")

        # Create tasks for all document grading operations
        tasks = [self.grade_document_async(doc, question) for doc in documents]

        # Execute all grading tasks concurrently
        grade_results = await asyncio.gather(*tasks, return_exceptions=True)

        relevant_docs = []
        for i, result in enumerate(grade_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Error grading document {documents[i].metadata.get('title', 'Unknown')}: {str(result)}"
                )
                # In case of error, include the document to be safe
                relevant_docs.append(documents[i])
            elif result.binary_score == "yes":
                relevant_docs.append(documents[i])
                logger.debug(
                    f"Document marked as relevant: {documents[i].metadata.get('title', 'Unknown')}"
                )
            else:
                logger.debug(
                    f"Document marked as irrelevant: {documents[i].metadata.get('title', 'Unknown')}"
                )

        logger.info(
            f"Filtered to {len(relevant_docs)} relevant documents out of {len(documents)}"
        )
        return relevant_docs

    def filter_relevant_documents_with_confidence(
        self, documents: List[Document], question: str
    ) -> tuple[List[Document], float]:
        """Filter documents based on relevance to the question and return confidence score."""
        logger.info(f"Grading {len(documents)} documents for relevance...")
        relevant_docs = []
        total_docs = len(documents)

        for doc in documents:
            try:
                grade_result = self.grade_document(doc, question)
                if grade_result.binary_score == "yes":
                    relevant_docs.append(doc)
                    logger.debug(
                        f"Document marked as relevant: {doc.metadata.get('title', 'Unknown')}"
                    )
                else:
                    logger.debug(
                        f"Document marked as irrelevant: {doc.metadata.get('title', 'Unknown')}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error grading document {doc.metadata.get('title', 'Unknown')}: {str(e)}"
                )
                # In case of error, include the document to be safe
                relevant_docs.append(doc)

        # Calculate confidence score
        confidence_score = len(relevant_docs) / total_docs if total_docs > 0 else 0.0
        logger.info(
            f"Filtered to {len(relevant_docs)} relevant documents out of {total_docs} (confidence: {confidence_score:.2f})"
        )
        return relevant_docs, confidence_score

    async def filter_relevant_documents_with_confidence_async(
        self, documents: List[Document], question: str
    ) -> tuple[List[Document], float]:
        """Async version of filter_relevant_documents_with_confidence for concurrent processing."""
        logger.info(f"Grading {len(documents)} documents for relevance (async)...")

        # Create tasks for all document grading operations
        tasks = [self.grade_document_async(doc, question) for doc in documents]

        # Execute all grading tasks concurrently
        grade_results = await asyncio.gather(*tasks, return_exceptions=True)

        relevant_docs = []
        total_docs = len(documents)

        for i, result in enumerate(grade_results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Error grading document {documents[i].metadata.get('title', 'Unknown')}: {str(result)}"
                )
                # In case of error, include the document to be safe
                relevant_docs.append(documents[i])
            elif result.binary_score == "yes":
                relevant_docs.append(documents[i])
                logger.debug(
                    f"Document marked as relevant: {documents[i].metadata.get('title', 'Unknown')}"
                )
            else:
                logger.debug(
                    f"Document marked as irrelevant: {documents[i].metadata.get('title', 'Unknown')}"
                )

        # Calculate confidence score
        confidence_score = len(relevant_docs) / total_docs if total_docs > 0 else 0.0
        logger.info(
            f"Filtered to {len(relevant_docs)} relevant documents out of {total_docs} (confidence: {confidence_score:.2f})"
        )
        return relevant_docs, confidence_score


class ReliableRAGWithConvoRetAndQueryAtomizationAndFewShot:
    """
    Unified RAG pipeline supporting both conversational memory and query atomization.
    """

    def __init__(
        self,
        experiment_name: str = "simple_rag_convo",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        temperature: float = DEFAULT_TEMPERATURE,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        debug: bool = False,
        highlighting_enabled: bool = False,
        search_type: str = DEFAULT_SEARCH_TYPE,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = DEFAULT_LAMBDA_MULT,
        max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
        semantic_chunking_threshold_type: str = "standard_deviation",
        semantic_chunking_threshold_amount: Optional[float] = None,
        use_semantic_chunking: bool = True,
        query_atomization_enabled: bool = True,
        use_few_shot: bool = True,
        num_examples: int = DEFAULT_NUM_EXAMPLES,
        document_grading_enabled: bool = True,
        show_confidence_level: bool = False,
    ):
        self.experiment_name = experiment_name
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model,
            breakpoint_threshold_type=semantic_chunking_threshold_type,
            breakpoint_threshold_amount=semantic_chunking_threshold_amount,
            use_semantic_chunking=use_semantic_chunking,
        )
        self.vector_store_builder = VectorStoreBuilder(
            embedding_model,
            collection_name="simple_rag_v1",
            persist_directory=persist_directory,
        )
        self.response_generator = ResponseGenerator(
            llm_model, temperature, max_history_turns, use_few_shot, num_examples
        )
        self.document_highlighter = DocumentHighlighter(
            llm_model, temperature, max_history_turns
        )
        self.document_grader = DocumentGrader(llm_model, temperature)
        self.retrieval_k = retrieval_k
        self.highlighting_enabled = highlighting_enabled
        self.document_grading_enabled = document_grading_enabled
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.max_history_turns = max_history_turns
        self.query_atomization_enabled = query_atomization_enabled
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples
        self.show_confidence_level = show_confidence_level

        self.query_contextualizer = QueryContextualizer(
            llm_model, temperature, max_history_turns
        )
        self.query_atomizer = QueryAtomizer(llm_model, temperature)
        self.debug = debug

        self.config = {
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_k": retrieval_k,
            "temperature": temperature,
            "persist_directory": persist_directory,
            "debug": debug,
            "highlighting_enabled": highlighting_enabled,
            "document_grading_enabled": document_grading_enabled,
            "search_type": search_type,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "max_history_turns": max_history_turns,
            "semantic_chunking_threshold_type": semantic_chunking_threshold_type,
            "semantic_chunking_threshold_amount": semantic_chunking_threshold_amount,
            "use_semantic_chunking": use_semantic_chunking,
            "query_atomization_enabled": query_atomization_enabled,
            "use_few_shot": use_few_shot,
            "num_examples": num_examples,
            "show_confidence_level": show_confidence_level,
        }

        self.metrics = {}
        self.documents = None
        self.document_chunks = None
        self.retriever = None
        self.last_question = None
        self.last_contextualized_question = None
        self.last_atomized_queries = None
        self.last_retrieved_docs = None
        self.last_generation = None
        self.last_highlights = None
        self.last_confidence_score = None

        self.chat_history = ChatMessageHistory()
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    def load_documents(self, pdf_dir: str = "data/pdfs") -> List[Document]:
        logger.info(f"Loading documents from {pdf_dir}...")
        self.documents = self.document_loader.load_pdf_documents(pdf_dir)
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents

    def process_documents(self) -> List[Document]:
        if self.documents is None:
            logger.error("Documents not loaded. Call load_documents first.")
            raise ValueError("Documents not loaded. Call load_documents first.")
        logger.info(
            f"Processing documents with semantic chunking (method: {self.config['semantic_chunking_threshold_type']})"
        )
        self.document_chunks = self.document_processor.split_documents(self.documents)
        logger.info(f"Created {len(self.document_chunks)} document chunks")
        return self.document_chunks

    def build_index(self) -> None:
        if self.document_chunks is None:
            logger.error("Document chunks not created. Call process_documents first.")
            raise ValueError(
                "Document chunks not created. Call process_documents first."
            )
        logger.info(
            f"Building index with embedding model {self.config['embedding_model']}..."
        )
        self.vector_store_builder.build_vectorstore(self.document_chunks)
        self.retriever = self.vector_store_builder.get_retriever(
            search_type=self.search_type,
            k=self.retrieval_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        return self.retriever

    def load_existing_vectorstore(self) -> None:
        logger.info("Loading existing vector store...")
        self.vector_store_builder.load_existing_vectorstore()
        self.retriever = self.vector_store_builder.get_retriever(
            search_type=self.search_type,
            k=self.retrieval_k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )
        logger.info(
            f"Retriever configured with search_type={self.search_type}, k={self.retrieval_k}"
        )
        return self.retriever

    def retrieve_documents(self, question: str) -> tuple[List[Document], float]:
        if self.retriever is None:
            logger.error("Retriever not created.")
            raise ValueError(
                "Retriever not created. Call build_index or load_existing_vectorstore first."
            )
        # Contextualize
        if self.chat_history.messages:
            contextualized_question = self.query_contextualizer.contextualize_query(
                question, self.chat_history.messages
            )
        else:
            contextualized_question = question
            logger.info(f"No chat history to contextualize question: {question}")
        self.last_question = question
        self.last_contextualized_question = contextualized_question

        # Atomize if enabled
        if self.query_atomization_enabled:
            atomized_queries = self.query_atomizer.atomize_query(
                contextualized_question
            )
            self.last_atomized_queries = atomized_queries
            logger.info(f"Atomized queries: {atomized_queries}")
            retrieved_docs = []
            for sub_query in atomized_queries:
                docs = self.retriever.invoke(sub_query)
                retrieved_docs.extend(docs)
            # Deduplicate
            seen_docs = set()
            unique_docs = []
            for doc in retrieved_docs:
                doc_key = (doc.page_content, doc.metadata.get("title", ""))
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_docs.append(doc)
            self.last_retrieved_docs = unique_docs
            logger.info(
                f"Retrieved {len(unique_docs)} unique documents from {len(atomized_queries)} sub-queries"
            )
        else:
            self.last_retrieved_docs = self.retriever.invoke(contextualized_question)
            logger.info(f"Retrieved {len(self.last_retrieved_docs)} documents")

        # Apply document grading if enabled
        if self.document_grading_enabled and self.last_retrieved_docs:
            logger.info("Applying document relevance grading...")
            self.last_retrieved_docs, confidence_score = (
                self.document_grader.filter_relevant_documents_with_confidence(
                    self.last_retrieved_docs, contextualized_question
                )
            )
            logger.info(
                f"After grading: {len(self.last_retrieved_docs)} relevant documents (confidence: {confidence_score:.2f})"
            )
        else:
            logger.info("Document grading is disabled")
            confidence_score = 1.0  # Assume 100% confidence when grading is disabled

        self.last_confidence_score = confidence_score
        return self.last_retrieved_docs, confidence_score

    async def retrieve_documents_async(
        self, question: str
    ) -> tuple[List[Document], float]:
        """Async version of retrieve_documents for concurrent processing."""
        if self.retriever is None:
            logger.error("Retriever not created.")
            raise ValueError(
                "Retriever not created. Call build_index or load_existing_vectorstore first."
            )

        # Contextualize (async)
        if self.chat_history.messages:
            contextualized_question = (
                await self.query_contextualizer.contextualize_query_async(
                    question, self.chat_history.messages
                )
            )
        else:
            contextualized_question = question
            logger.info(f"No chat history to contextualize question: {question}")
        self.last_question = question
        self.last_contextualized_question = contextualized_question

        # Atomize if enabled (async)
        if self.query_atomization_enabled:
            atomized_queries = await self.query_atomizer.atomize_query_async(
                contextualized_question
            )
            self.last_atomized_queries = atomized_queries
            logger.info(f"Atomized queries: {atomized_queries}")

            # Concurrent retrieval for all sub-queries
            retrieval_tasks = [
                self.retriever.ainvoke(sub_query) for sub_query in atomized_queries
            ]
            retrieved_docs_lists = await asyncio.gather(
                *retrieval_tasks, return_exceptions=True
            )

            # Combine and deduplicate results
            retrieved_docs = []
            for docs_list in retrieved_docs_lists:
                if isinstance(docs_list, Exception):
                    logger.warning(f"Error in sub-query retrieval: {str(docs_list)}")
                    continue
                retrieved_docs.extend(docs_list)

            # Deduplicate
            seen_docs = set()
            unique_docs = []
            for doc in retrieved_docs:
                doc_key = (doc.page_content, doc.metadata.get("title", ""))
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_docs.append(doc)
            self.last_retrieved_docs = unique_docs
            logger.info(
                f"Retrieved {len(unique_docs)} unique documents from {len(atomized_queries)} sub-queries (async)"
            )
        else:
            self.last_retrieved_docs = await self.retriever.ainvoke(
                contextualized_question
            )
            logger.info(f"Retrieved {len(self.last_retrieved_docs)} documents (async)")

        # Apply document grading if enabled (async)
        if self.document_grading_enabled and self.last_retrieved_docs:
            logger.info("Applying document relevance grading (async)...")
            self.last_retrieved_docs, confidence_score = (
                await self.document_grader.filter_relevant_documents_with_confidence_async(
                    self.last_retrieved_docs, contextualized_question
                )
            )
            logger.info(
                f"After grading: {len(self.last_retrieved_docs)} relevant documents (confidence: {confidence_score:.2f})"
            )
        else:
            logger.info("Document grading is disabled")
            confidence_score = 1.0  # Assume 100% confidence when grading is disabled

        self.last_confidence_score = confidence_score
        return self.last_retrieved_docs, confidence_score

    def generate_response(self) -> str:
        if self.last_retrieved_docs is None:
            logger.error("No documents retrieved. Call retrieve_documents first.")
            raise ValueError("No documents retrieved. Call retrieve_documents first.")
        logger.info(
            f"Generating response with model {self.config['llm_model']} (temperature={self.config['temperature']})"
        )
        self.last_generation = self.response_generator.generate_response(
            self.last_retrieved_docs,
            self.last_contextualized_question,
            self.chat_history.messages,
        )

        # Add confidence level information if enabled
        if self.show_confidence_level and self.last_confidence_score is not None:
            confidence_percentage = self.last_confidence_score * 100
            confidence_level = self._get_confidence_level_description(
                self.last_confidence_score
            )
            confidence_info = f"\n\n[Confidence Level: {confidence_level} ({confidence_percentage:.1f}%)]"
            self.last_generation += confidence_info

        return self.last_generation

    def _get_confidence_level_description(self, confidence_score: float) -> str:
        """Convert confidence score to descriptive level."""
        if confidence_score >= 0.9:
            return "Very High"
        elif confidence_score >= 0.7:
            return "High"
        elif confidence_score >= 0.5:
            return "Medium"
        elif confidence_score >= 0.3:
            return "Low"
        else:
            return "Very Low"

    def highlight_documents(self) -> HighlightDocuments:
        if self.last_generation is None:
            logger.error("No response generated. Call generate_response first.")
            raise ValueError("No response generated. Call generate_response first.")
        self.last_highlights = self.document_highlighter.highlight_documents(
            self.last_retrieved_docs,
            self.last_contextualized_question,
            self.last_generation,
            self.chat_history.messages,
        )
        logger.info(
            f"Highlighted {len(self.last_highlights.segment)} document segments"
        )
        return self.last_highlights

    def run_complete_pipeline(
        self,
        question: str,
        min_confidence_threshold: float = 0.5,
        skip_logging: bool = False,
    ) -> Dict[str, Any]:
        result = {
            "question": question,
            "retrieved_docs": [],
            "response": "",
            "highlights": None,
            "run_id": None,
            "errors": [],
            "confidence_score": 0.0,
        }
        try:
            self.chat_history.add_message(HumanMessage(content=question))
            logger.debug(f"Processing question '{question}'")
            try:
                result["retrieved_docs"], result["confidence_score"] = (
                    self.retrieve_documents(question)
                )
                logger.info(
                    f"Retrieval confidence score: {result['confidence_score']:.2f}"
                )
            except Exception as e:
                error_msg = f"Error retrieving documents: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                return result

            # Check confidence threshold
            if result["confidence_score"] < min_confidence_threshold:
                logger.warning(
                    f"Confidence score {result['confidence_score']:.2f} below threshold {min_confidence_threshold}, returning fallback response"
                )
                fallback_response = "Sorry, I wasn't able to find sufficient information to confidently answer your question."

                # Add confidence level information if enabled
                if self.show_confidence_level:
                    confidence_percentage = result["confidence_score"] * 100
                    confidence_level = self._get_confidence_level_description(
                        result["confidence_score"]
                    )
                    confidence_info = f"\n\n[Confidence Level: {confidence_level} ({confidence_percentage:.1f}%) - Below threshold]"
                    fallback_response += confidence_info

                result["response"] = fallback_response
                self.chat_history.add_message(AIMessage(content=fallback_response))
                return result

            try:
                result["response"] = self.generate_response()
                logger.debug(
                    f"Successfully generated response: {result['response'][:100]}..."
                )
                self.chat_history.add_message(AIMessage(content=result["response"]))
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                fallback_response = "Sorry, I couldn't generate a response based on the retrieved information."
                result["response"] = fallback_response
                self.chat_history.add_message(AIMessage(content=fallback_response))
            if self.highlighting_enabled:
                try:
                    result["highlights"] = self.highlight_documents()
                    logger.debug(
                        f"Successfully highlighted {len(result['highlights'].segment) if result['highlights'] else 0} segments"
                    )
                except Exception as e:
                    error_msg = f"Error highlighting documents: {str(e)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
            else:
                logger.info("Document highlighting is disabled")
        except Exception as e:
            error_msg = f"Unexpected error in RAG pipeline: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
        return result

    async def run_complete_pipeline_async(
        self,
        question: str,
        min_confidence_threshold: float = 0.5,
        skip_logging: bool = False,
    ) -> Dict[str, Any]:
        """Async version of run_complete_pipeline for concurrent processing."""
        result = {
            "question": question,
            "retrieved_docs": [],
            "response": "",
            "highlights": None,
            "run_id": None,
            "errors": [],
            "confidence_score": 0.0,
        }
        try:
            self.chat_history.add_message(HumanMessage(content=question))
            logger.debug(f"Processing question '{question}' (async)")
            try:
                result["retrieved_docs"], result["confidence_score"] = (
                    await self.retrieve_documents_async(question)
                )
                logger.info(
                    f"Retrieval confidence score: {result['confidence_score']:.2f}"
                )
            except Exception as e:
                error_msg = f"Error retrieving documents: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                return result

            # Check confidence threshold
            if result["confidence_score"] < min_confidence_threshold:
                logger.warning(
                    f"Confidence score {result['confidence_score']:.2f} below threshold {min_confidence_threshold}, returning fallback response"
                )
                fallback_response = "Sorry, I wasn't able to find sufficient information to confidently answer your question."

                # Add confidence level information if enabled
                if self.show_confidence_level:
                    confidence_percentage = result["confidence_score"] * 100
                    confidence_level = self._get_confidence_level_description(
                        result["confidence_score"]
                    )
                    confidence_info = f"\n\n[Confidence Level: {confidence_level} ({confidence_percentage:.1f}%) - Below threshold]"
                    fallback_response += confidence_info

                result["response"] = fallback_response
                self.chat_history.add_message(AIMessage(content=fallback_response))
                return result

            try:
                result["response"] = self.generate_response()
                logger.debug(
                    f"Successfully generated response: {result['response'][:100]}..."
                )
                self.chat_history.add_message(AIMessage(content=result["response"]))
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                fallback_response = "Sorry, I couldn't generate a response based on the retrieved information."
                result["response"] = fallback_response
                self.chat_history.add_message(AIMessage(content=fallback_response))
            if self.highlighting_enabled:
                try:
                    result["highlights"] = self.highlight_documents()
                    logger.debug(
                        f"Successfully highlighted {len(result['highlights'].segment) if result['highlights'] else 0} segments"
                    )
                except Exception as e:
                    error_msg = f"Error highlighting documents: {str(e)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
            else:
                logger.info("Document highlighting is disabled")
        except Exception as e:
            error_msg = f"Unexpected error in RAG pipeline: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
        return result

    def get_chat_history(self) -> List:
        return self.chat_history.messages

    def clear_chat_history(self) -> None:
        logger.info("Clearing chat history")
        self.chat_history.clear()

    def get_evaluation_samples(self) -> List[Dict[str, Any]]:
        if self.last_question is None:
            logger.error("No question found. Call run_complete_pipeline first.")
            raise ValueError("No question found. Call run_complete_pipeline first.")
        contexts = []
        if self.last_retrieved_docs:
            contexts = [doc.page_content for doc in self.last_retrieved_docs]
        response = ""
        if self.last_generation:
            response = self.last_generation
        sample = {
            "user_input": self.last_contextualized_question,
            "response": response,
            "reference": "",
            "retrieved_contexts": contexts,
        }
        logger.debug(
            f"Created evaluation sample for question: {self.last_contextualized_question}"
        )
        return [sample]

    def run_complete_pipeline_streaming(
        self,
        question: str,
        min_confidence_threshold: float = 0.2,
        skip_logging: bool = False,
    ):
        """Run the pipeline and yield tokens for streaming response."""
        try:
            self.chat_history.add_message(HumanMessage(content=question))
            try:
                retrieved_docs, confidence_score = self.retrieve_documents(question)
            except Exception as e:
                error_msg = f"[Error retrieving documents: {str(e)}]"
                for char in error_msg:
                    yield char
                return

            # Check confidence threshold
            if confidence_score < min_confidence_threshold:
                fallback_response = "Sorry, I wasn't able to find sufficient information to confidently answer your question."
                # Stream the fallback response character by character
                for char in fallback_response:
                    yield char
                self.chat_history.add_message(AIMessage(content=fallback_response))
                return

            # Streaming response
            try:
                stream = self.response_generator.generate_streaming_response(
                    retrieved_docs,
                    self.last_contextualized_question,
                    self.chat_history.messages,
                )
                full_response = ""
                for chunk in stream:
                    token = getattr(chunk, "content", None)
                    if token is None:
                        token = str(chunk)
                    full_response += token
                    yield token
                self.chat_history.add_message(AIMessage(content=full_response))
            except Exception as e:
                fallback_response = f"[Error generating response: {str(e)}]"
                # Stream the error response character by character
                for char in fallback_response:
                    yield char
                self.chat_history.add_message(AIMessage(content=fallback_response))
        except Exception as e:
            error_msg = f"[Unexpected error in RAG pipeline: {str(e)}]"
            # Stream the error response character by character
            for char in error_msg:
                yield char

    async def run_complete_pipeline_streaming_async(
        self,
        question: str,
        min_confidence_threshold: float = 0.2,
        skip_logging: bool = False,
    ):
        """Async version of run_complete_pipeline_streaming for concurrent processing."""
        try:
            self.chat_history.add_message(HumanMessage(content=question))
            try:
                retrieved_docs, confidence_score = await self.retrieve_documents_async(
                    question
                )
            except Exception as e:
                error_msg = f"[Error retrieving documents: {str(e)}]"
                for char in error_msg:
                    yield char
                return

            # Check confidence threshold
            if confidence_score < min_confidence_threshold:
                fallback_response = "Sorry, I wasn't able to find sufficient information to confidently answer your question."
                # Stream the fallback response character by character
                for char in fallback_response:
                    yield char
                self.chat_history.add_message(AIMessage(content=fallback_response))
                return

            # Streaming response
            try:
                stream = self.response_generator.generate_streaming_response(
                    retrieved_docs,
                    self.last_contextualized_question,
                    self.chat_history.messages,
                )
                full_response = ""
                for chunk in stream:
                    token = getattr(chunk, "content", None)
                    if token is None:
                        token = str(chunk)
                    full_response += token
                    yield token
                self.chat_history.add_message(AIMessage(content=full_response))
            except Exception as e:
                fallback_response = f"[Error generating response: {str(e)}]"
                # Stream the error response character by character
                for char in fallback_response:
                    yield char
                self.chat_history.add_message(AIMessage(content=fallback_response))
        except Exception as e:
            error_msg = f"[Unexpected error in RAG pipeline: {str(e)}]"
            # Stream the error response character by character
            for char in error_msg:
                yield char
