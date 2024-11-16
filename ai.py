import os
from typing import Dict, List
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from dotenv import load_dotenv

# Initialize OpenTelemetry instrumentation
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

tracer_provider = register(
    project_name="paper2code",
    endpoint="http://localhost:6006/v1/traces",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize the LLM
# llm = ChatOpenAI(
#     model_name="gpt-4",
#     temperature=0.7,
#     streaming=False,
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Define output schemas
architecture_schema = ResponseSchema(
    name="architecture",
    description="System architecture including components and their interactions",
)
dependencies_schema = ResponseSchema(
    name="dependencies",
    description="List of required Python libraries and dependencies",
)
modules_schema = ResponseSchema(
    name="modules", description="Description of each module/component needed"
)

architecture_parser = StructuredOutputParser.from_response_schemas(
    [architecture_schema, dependencies_schema, modules_schema]
)

# Define prompts with structured output
architecture_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """Analyze the paper and extract the system architecture, required dependencies, and modules.
    {format_instructions}"""
        ),
        HumanMessagePromptTemplate.from_template(
            """Input text: {input_text}, Context: {context}"""
        ),
    ]
)

architecture_documentation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a documentation generator that produces clear, concise documentation based on the given requirements. 
            Requirements for the generated documentation:
            1. Include a brief description of the file's purpose.
            2. List all functions and classes with their signatures and descriptions.
            3. Provide a summary of the dependencies required for the file.
            4. Follow the Google Python Style Guide for docstrings.
            5. Include only the documentation text, without any code implementation.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """Filename: {filename}
            Architecture: {architecture}
            Context: {context}
    Purpose: {purpose}
    Functions: {functions}
    Dependencies: {dependencies}"""
        ),
    ]
)

file_structure_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """Based on the architecture and modules, define the Python files needed.
    Each file should include:
    - filename
    - purpose
    - list of functions with their signatures and descriptions
    
    Format as JSON with the structure:
    {{
        "files": [
            {{
                "filename": "example.py",
                "purpose": "description",
                "functions": [
                    {{
                        "name": "function_name",
                        "signature": "def function_name(param1: type) -> return_type",
                        "description": "what the function does"
                    }}
                ]
            }}
        ]
    }}"""
        ),
        HumanMessagePromptTemplate.from_template(
            """Architecture: {architecture}
    Modules: {modules}"""
        ),
    ]
)

code_generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """ You are a Python code generator that produces complete, functional Python code based on the given requirements. 
            Requirements for the generated code:
            1. Include all necessary `import` statements based on the specified dependencies
            2. Follow PEP 8 style guidelines strictly
            3. Provide clear, concise docstrings for the file, each function, and any classes.
            4. Use type hints for all function parameters and return types.
            5. Ensure all functions listed are implemented with the specified behavior.
            6. Structure the code logically, starting with imports, followed by global variables (if any), and then functions/classes.
            7. Output the Python code only. Comments within the code are acceptable, but no markdown, external formatting, or any additional text outside the code block should be included.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """Filename: {filename}
    Purpose: {purpose}
    Functions: {functions}
    Dependencies: {dependencies}"""
        ),
    ]
)

# documentation_generation_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template(
#             """You are a documentation generator that produces clear, concise documentation based on the given requirements.
#             Requirements for the generated documentation:
#             1. Include a brief description of the file's purpose.
#             2. List all functions and classes with their signatures and descriptions.
#             3. Provide a summary of the dependencies required for the file.
#             4. Follow the Google Python Style Guide for docstrings.
#             5. Include only the documentation text, without any code implementation.
#             """
#         ),
#         HumanMessagePromptTemplate.from_template(
#             """Filename: {filename}
#     Purpose: {purpose}
#     Functions: {functions}
#     Dependencies: {dependencies}"""
#         ),
#     ]
# )


def create_repository(
    paper_text: str, progress, output_dir: str = "generated_repo"
) -> None:
    """
    Create a Python repository from a research paper.

    Args:
        paper_text (str): The text content of the research paper
        output_dir (str): Directory where the code will be generated
    """
    progress("Starting", 1)

    if output_dir == "generated_repo":
        output_dir = f"generated_repo_output/{uuid4().hex[:6]}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split text and create FAISS index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(paper_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    progress("Indexing complete", 2)
    # Add reranking for better context retrieval
    compressor = FlashrankRerank()
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    progress("Context retrieval ready", 3)
    # Step 1: Analyze architecture with context
    arch_context = retriever.get_relevant_documents(
        "system architecture components dependencies modules"
    )
    arch_context_text = "\n".join([doc.page_content for doc in arch_context])

    # Create the chain using LCEL pipe syntax
    chain = architecture_prompt | llm | architecture_parser

    paper_text_input = paper_text
    # if len(paper_text) > 4096:
    #     paper_text_input = paper_text[:4096]

    # Invoke the chain with all required variables
    architecture_output = chain.invoke(
        {
            "input_text": paper_text_input,
            "context": arch_context_text,
            "format_instructions": architecture_parser.get_format_instructions(),
        }
    )
    progress("Architecture analysis complete", 4)

    # Step 2: Define file structure with context
    file_structure_chain = file_structure_prompt | llm | JsonOutputParser()
    progress("Generating file structure", 5)

    # Get relevant context for file structure
    structure_context = retriever.get_relevant_documents(
        "code organization file structure implementation details"
    )
    structure_context_text = "\n".join([doc.page_content for doc in structure_context])

    progress("File structure ready", 6)
    file_structure = file_structure_chain.invoke(
        {
            "architecture": architecture_output["architecture"],
            "modules": architecture_output["modules"],
            "context": structure_context_text,
        }
    )
    progress("File structure generated", 7)

    # Step 3: Generate code for each file with relevant context
    code_generation_chain = code_generation_prompt | llm

    count = 0
    for file_info in file_structure["files"]:
        # Get relevant context for this specific file
        file_context = retriever.get_relevant_documents(
            f"{file_info['purpose']} {' '.join([f['description'] for f in file_info['functions']])}"
        )
        file_context_text = "\n".join([doc.page_content for doc in file_context])

        code = code_generation_chain.invoke(
            input={
                "filename": file_info["filename"],
                "purpose": file_info["purpose"],
                "functions": file_info["functions"],
                "dependencies": architecture_output["dependencies"],
                "context": file_context_text,
            }
        )
        count += 1
        progress(f"Generated {file_info['filename']}", 7 + count)
        code_out = code.content
        code_out = code_out.strip()
        # Remove "```python" and "```" from only the beginning and end
        if code_out.startswith("```python"):
            code_out = code_out[len("```python") :].strip()
        if code_out.endswith("```"):
            code_out = code_out[: -len("```")].strip()
        # Write code to file
        file_path = os.path.join(output_dir, file_info["filename"])
        with open(file_path, "w") as f:
            f.write(code_out)

        print(f"Generated {file_info['filename']}")

    # Step 4: Generate documentation for the overall architecture
    documentation_chain = architecture_documentation_prompt | llm
    documentation_output = documentation_chain.invoke(
        {
            "filename": "README.md",
            "architecture": architecture_output["architecture"],
            "context": arch_context_text,
            "purpose": "Documentation for the system architecture",
            "functions": [],
            "dependencies": architecture_output["dependencies"],
        }
    )

    # Write documentation to file
    doc_path = os.path.join(output_dir, "README.md")
    with open(doc_path, "w") as f:
        f.write(documentation_output.content)


if __name__ == "__main__":
    # Example usage
    paper_path = "./text.md"
    with open(paper_path, "r") as file:
        paper_text = file.read()
        # Print the first 1000 characters
        print(paper_text[:1000])

        def progress(msg, percent):
            print(msg, percent)

        create_repository(paper_text, progress)
