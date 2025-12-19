import gradio as gr
import os
import tempfile
from pathlib import Path
import time
from typing import List, Optional, Tuple, Dict, Any
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Environment and config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AdvancedGradioRAG:
    def __init__(self):
        self.vectorstores = {}
        self.qa_chains = {}
        self.embeddings = {}
        self.llm_models = {}
        self.performance_metrics = []
        self.processed_files = []
        self.current_vectorstore_type = "chroma"
        self.setup_components()
    
    def setup_components(self):
        """Initialize advanced components"""
        try:
            # Initialize embeddings
            self.embeddings = {
                "fast": HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                ),
                "accurate": HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            }
            
            # Initialize multiple Groq models
            self.llm_models = {
                "OPENAI": ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=os.getenv("GROQ_API_KEY_VS"),
                    temperature=0.1,
                    max_tokens=1024
                ),
                "QWEN": ChatGroq(
                    model="qwen/qwen3-32b",
                    api_key=os.getenv("GROQ_API_KEY_VS"),
                    temperature=0.1,
                    max_tokens=1024
                ),
                "KIMI": ChatGroq(
                    model="moonshotai/kimi-k2-instruct-0905",
                    api_key=os.getenv("GROQ_API_KEY_VS"),
                    temperature=0.1,
                    max_tokens=1024
                ),
                "LLAMA": ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=os.getenv("GROQ_API_KEY_VS"),
                    temperature=0.1,
                    max_tokens=1024
                ),
                "GROQ": ChatGroq(
                    model="groq/compound",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.1,
                    max_tokens=1024
                )
            }
            
            # Initialize vector stores
            self.setup_vectorstores()
            return True
            
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            return False
    
    def setup_vectorstores(self):
        """Setup multiple vector stores"""
        embedding_func = self.embeddings["fast"]
        
        # ChromaDB
        self.vectorstores["chroma"] = Chroma(
            persist_directory="./chroma_db_gradio_advanced",
            embedding_function=embedding_func
        )
        
        # FAISS (with error handling)
        try:
            if os.path.exists("./faiss_index_gradio"):
                self.vectorstores["faiss"] = FAISS.load_local(
                    "./faiss_index_gradio", 
                    embedding_func,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create empty FAISS index
                sample_text = ["Sample initialization text"]
                sample_embeddings = embedding_func.embed_documents(sample_text)
                self.vectorstores["faiss"] = FAISS.from_embeddings(
                    [(sample_text[0], sample_embeddings[0])],
                    embedding_func
                )
        except Exception as e:
            print(f"FAISS initialization warning: {e}")
    
    def setup_qa_chain(self, model_name: str = "llama3-8b", use_compression: bool = False):
        """Setup QA chain with advanced options"""
        vectorstore = self.vectorstores[self.current_vectorstore_type]
        
        # Check if vectorstore has documents
        try:
            if self.current_vectorstore_type == "chroma":
                has_docs = hasattr(vectorstore, '_collection') and vectorstore._collection.count() > 0
            elif self.current_vectorstore_type == "faiss":
                has_docs = hasattr(vectorstore, 'docstore') and len(vectorstore.docstore._dict) > 1  # >1 because of init doc
            else:
                has_docs = False
        except Exception:
            has_docs = False
        
        if not has_docs:
            return
        
        advanced_prompt = PromptTemplate(
            template="""You are an expert assistant analyzing documents. Provide comprehensive and accurate answers.

            Guidelines:
            1. Answer based on the provided context
            2. Be specific and cite relevant information
            3. If information is incomplete, state what's missing
            4. Provide detailed analysis when requested
            
            Context: {context}
            
            Question: {question}
            
            Detailed Answer:""",
            input_variables=["context", "question"]
        )
        
        try:
            # Setup retriever
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 20,
                    "lambda_mult": 0.7
                }
            )
            
            # Optional compression
            if use_compression and self.llm_models.get(model_name):
                compressor = LLMChainExtractor.from_llm(self.llm_models[model_name])
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever
                )
            
            # Create QA chain
            chain_key = f"{model_name}_{self.current_vectorstore_type}_{'compressed' if use_compression else 'standard'}"
            
            self.qa_chains[chain_key] = RetrievalQA.from_chain_type(
                llm=self.llm_models[model_name],
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": advanced_prompt},
                return_source_documents=True
            )
        except Exception as e:
            print(f"Error setting up QA chain: {e}")
    
    def process_documents_advanced(self, files, chunk_strategy: str = "standard"):
        """Advanced document processing"""
        if not files:
            return "No files uploaded.", "", self.get_status_dict()
        
        try:
            chunk_strategies = {
                "standard": {"chunk_size": 1000, "overlap": 200},
                "small": {"chunk_size": 500, "overlap": 100},
                "large": {"chunk_size": 1500, "overlap": 300},
                "semantic": {"chunk_size": 1000, "overlap": 200, "separators": ["\n\n", "\n", ". ", " "]}
            }
            
            strategy_config = chunk_strategies.get(chunk_strategy, chunk_strategies["standard"])
            
            all_documents = []
            file_info = []
            start_time = time.time()
            
            for file_path in files:
                filename = os.path.basename(file_path)
                
                try:
                    if filename.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif filename.endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                    else:
                        continue
                    
                    documents = loader.load()
                    
                    for doc in documents:
                        doc.metadata.update({
                            'source': filename,
                            'upload_time': datetime.now().isoformat(),
                            'chunk_strategy': chunk_strategy
                        })
                    
                    all_documents.extend(documents)
                    file_info.append(f"✅ {filename}: {len(documents)} pages")
                    
                except Exception as e:
                    file_info.append(f"❌ {filename}: Error - {str(e)}")
            
            if not all_documents:
                return "No valid documents processed.", "\n".join(file_info), self.get_status_dict()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=strategy_config["chunk_size"],
                chunk_overlap=strategy_config["overlap"],
                length_function=len,
                separators=strategy_config.get("separators", ["\n\n", "\n", " ", ""])
            )
            
            split_documents = text_splitter.split_documents(all_documents)
            
            # Add to both vector stores
            for vs_name, vs in self.vectorstores.items():
                try:
                    if vs_name == "chroma":
                        vs.add_documents(split_documents)
                        vs.persist()
                    elif vs_name == "faiss":
                        vs.add_documents(split_documents)
                        vs.save_local("./faiss_index_gradio")
                except Exception as e:
                    print(f"Error adding to {vs_name}: {e}")
            
            # Update processed files
            self.processed_files.extend([os.path.basename(f) for f in files])
            
            # Setup QA chains for all models
            for model_name in self.llm_models.keys():
                self.setup_qa_chain(model_name)
                self.setup_qa_chain(model_name, use_compression=True)
            
            processing_time = time.time() - start_time
            
            status = f"🎉 Successfully processed {len(files)} files, created {len(split_documents)} chunks in {processing_time:.2f}s!"
            details = "\n".join(file_info)
            
            return status, details, self.get_status_dict()
            
        except Exception as e:
            return f"❌ Error processing files: {str(e)}", "", self.get_status_dict()
    
    def ask_question_advanced(
        self, 
        question: str, 
        history: List, 
        model_name: str = "llama3-8b",
        vectorstore_type: str = "chroma",
        use_compression: bool = False
    ):
        """Advanced question answering"""
        if not question.strip():
            return history, ""
        
        self.current_vectorstore_type = vectorstore_type
        chain_key = f"{model_name}_{vectorstore_type}_{'compressed' if use_compression else 'standard'}"
        
        if chain_key not in self.qa_chains:
            self.setup_qa_chain(model_name, use_compression)
        
        qa_chain = self.qa_chains.get(chain_key)
        if not qa_chain:
            response = "Please upload and process documents first."
            history.append([question, response])
            return history, ""
        
        try:
            start_time = time.time()
            result = qa_chain.invoke({"query": question})
            end_time = time.time()
            
            response_time = end_time - start_time
            answer = result['result']
            sources = result.get('source_documents', [])
            
            # Track performance
            metric = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "model": model_name,
                "vectorstore": vectorstore_type,
                "compression": use_compression,
                "response_time": response_time,
                "answer_length": len(answer),
                "num_sources": len(sources)
            }
            self.performance_metrics.append(metric)
            
            # Format response
            response = f"{answer}\n\n"
            response += f"⚡ **Performance**: {response_time:.2f}s | Model: {model_name} | Vector Store: {vectorstore_type}\n"
            
            if sources:
                response += f"\n📚 **Sources** ({len(sources)} found):\n"
                for i, source in enumerate(sources[:3]):
                    source_name = source.metadata.get('source', 'Unknown')
                    preview = source.page_content[:150] + "..." if len(source.page_content) > 150 else source.page_content
                    response += f"{i+1}. **{source_name}**: {preview}\n\n"
            
            history.append([question, response])
            return history, ""
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([question, error_response])
            return history, ""
    
    def get_status_dict(self):
        """Get comprehensive status"""
        status = {}
        for vs_name, vs in self.vectorstores.items():
            try:
                if vs_name == "chroma":
                    count = vs._collection.count() if hasattr(vs, '_collection') else 0
                elif vs_name == "faiss":
                    count = len(vs.docstore._dict) - 1 if hasattr(vs, 'docstore') else 0  # -1 for init doc
                    count = max(0, count)
                else:
                    count = 0
                status[vs_name] = count
            except Exception:
                status[vs_name] = 0
        
        return status
    
    def get_performance_analytics(self):
        """Get performance analytics for dashboard"""
        if not self.performance_metrics:
            return None, "No performance data available yet."
        
        df = pd.DataFrame(self.performance_metrics)
        
        # Create performance plots
        try:
            # Response time over time
            fig_timeline = px.line(
                df, 
                x='timestamp', 
                y='response_time',
                color='model',
                title='Response Time Over Time',
                labels={'response_time': 'Response Time (s)'}
            )
            
            # Model comparison
            model_stats = df.groupby('model')['response_time'].agg(['mean', 'count']).reset_index()
            fig_models = px.bar(
                model_stats,
                x='model',
                y='mean',
                title='Average Response Time by Model',
                labels={'mean': 'Avg Response Time (s)'}
            )
            
            # Summary stats
            summary = {
                "total_queries": len(df),
                "avg_response_time": df['response_time'].mean(),
                "fastest_query": df['response_time'].min(),
                "slowest_query": df['response_time'].max()
            }
            
            return (fig_timeline, fig_models, summary), "Analytics updated successfully!"
            
        except Exception as e:
            return None, f"Error generating analytics: {str(e)}"
    
    def run_benchmark(self, vectorstore_type: str = "chroma"):
        """Run model benchmark"""
        self.current_vectorstore_type = vectorstore_type
        
        # Check if we have documents
        status = self.get_status_dict()
        if status.get(vectorstore_type, 0) == 0:
            return "No documents available for benchmarking. Please upload documents first."
        
        benchmark_question = "What are the main topics and key insights discussed in these documents?"
        results = []
        
        for model_name in self.llm_models.keys():
            try:
                chain_key = f"{model_name}_{vectorstore_type}_standard"
                if chain_key not in self.qa_chains:
                    self.setup_qa_chain(model_name)
                
                qa_chain = self.qa_chains.get(chain_key)
                if qa_chain:
                    start_time = time.time()
                    result = qa_chain.invoke({"query": benchmark_question})
                    end_time = time.time()
                    
                    results.append({
                        "Model": model_name,
                        "Response Time (s)": f"{end_time - start_time:.3f}",
                        "Answer Length": len(result['result']),
                        "Sources Found": len(result.get('source_documents', []))
                    })
                else:
                    results.append({
                        "Model": model_name,
                        "Response Time (s)": "Error",
                        "Answer Length": 0,
                        "Sources Found": 0
                    })
            except Exception as e:
                results.append({
                    "Model": model_name,
                    "Response Time (s)": f"Error: {str(e)[:30]}...",
                    "Answer Length": 0,
                    "Sources Found": 0
                })
        
        # Convert to display format
        if results:
            df = pd.DataFrame(results)
            fastest_model = "N/A"
            
            # Find fastest model
            try:
                numeric_times = []
                for idx, row in df.iterrows():
                    try:
                        time_val = float(row['Response Time (s)'])
                        numeric_times.append((row['Model'], time_val))
                    except Exception:
                        continue
                
                if numeric_times:
                    fastest_model = min(numeric_times, key=lambda x: x[1])[0]
            except Exception:
                pass
            
            benchmark_text = f"🏆 Fastest Model: {fastest_model}\n\n"
            benchmark_text += df.to_string(index=False)
            
            return benchmark_text
        else:
            return "Benchmark failed to run."
    
    def clear_database(self):
        """Clear all databases"""
        try:
            for vs_name, vs in self.vectorstores.items():
                if vs_name == "chroma" and hasattr(vs, 'delete_collection'):
                    vs.delete_collection()
                elif vs_name == "faiss" and hasattr(vs, 'docstore'):
                    vs.docstore._dict.clear()
            
            # Reinitialize
            self.setup_vectorstores()
            self.qa_chains.clear()
            self.processed_files = []
            self.performance_metrics = []
            
            return "🧹 All databases cleared successfully!", self.get_status_dict()
        except Exception as e:
            return f"❌ Error clearing databases: {str(e)}", self.get_status_dict()

# Initialize the advanced RAG assistant
advanced_rag = AdvancedGradioRAG()

def create_advanced_gradio_interface():
    """Create the advanced Gradio interface"""
    
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .metric-box {
        background: linear-gradient(45deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🚀 Advanced RAG Assistant", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🚀 Advanced Ultra-Fast RAG Assistant</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                <strong>Multi-Model • Multi-Vector Store • Real-time Analytics</strong>
            </p>
        </div>
        """)
        
        # Check API key
        if not os.getenv("GROQ_API_KEY"):
            gr.HTML("""
            <div style="background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 15px; margin: 20px 0;">
                <h3 style="color: #dc2626; margin: 0;">🔑 API Key Required</h3>
                <p style="color: #7f1d1d; margin: 5px 0 0 0;">
                    Please set your GROQ_API_KEY in your environment variables.<br>
                    Get your free API key from: <a href="https://console.groq.com/" target="_blank">console.groq.com</a>
                </p>
            </div>
            """)
            return demo
        
        # Main tabs
        with gr.Tabs():
            
            # Chat Tab
            with gr.TabItem("💬 Intelligent Chat"):
                with gr.Row():
                    # Left panel - Configuration
                    with gr.Column(scale=1):
                        gr.HTML("<h3>⚙️ Configuration</h3>")
                        
                        model_dropdown = gr.Dropdown(
                            choices=list(advanced_rag.llm_models.keys()),
                            value="llama3-8b",
                            label="🤖 Model Selection",
                            info="Choose your Groq model"
                        )
                        
                        vectorstore_dropdown = gr.Dropdown(
                            choices=["chroma", "faiss"],
                            value="chroma",
                            label="🗂️ Vector Store",
                            info="Select vector database"
                        )
                        
                        compression_checkbox = gr.Checkbox(
                            label="🗜️ Use Contextual Compression",
                            value=False,
                            info="Compress context for better relevance"
                        )
                        
                        # Status display
                        gr.HTML("<h3>📊 System Status</h3>")
                        status_json = gr.JSON(
                            label="Document Counts",
                            value=advanced_rag.get_status_dict()
                        )
                        
                    # Right panel - Chat
                    with gr.Column(scale=2):
                        gr.HTML("<h3>💬 Document Q&A</h3>")
                        
                        chatbot = gr.Chatbot(
                            label="AI Assistant",
                            height=600,
                            avatar_images=("🧑‍💻", "🤖"),
                            bubble_full_width=False
                        )
                        
                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask anything about your documents...",
                                scale=4
                            )
                            ask_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        # Quick actions
                        with gr.Row():
                            quick_btns = [
                                gr.Button("📋 Summarize All", size="sm"),
                                gr.Button("🔍 Find Methods", size="sm"),
                                gr.Button("💡 Key Insights", size="sm"),
                                gr.Button("📊 Extract Data", size="sm")
                            ]
            
            # Document Management Tab
            with gr.TabItem("📄 Document Management"):
                with gr.Row():
                    # Upload section
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📤 Upload Documents</h3>")
                        
                        file_upload = gr.File(
                            label="Select Files",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"]
                        )
                        
                        chunk_strategy = gr.Dropdown(
                            choices=["standard", "small", "large", "semantic"],
                            value="standard",
                            label="📝 Chunking Strategy",
                            info="How to split documents"
                        )
                        
                        process_btn = gr.Button(
                            "🔄 Process Documents",
                            variant="primary",
                            size="lg"
                        )
                        
                        clear_btn = gr.Button(
                            "🗑️ Clear All Databases",
                            variant="secondary"
                        )
                    
                    # Results section
                    with gr.Column(scale=2):
                        gr.HTML("<h3>📋 Processing Results</h3>")
                        
                        process_status = gr.Textbox(
                            label="Status",
                            lines=2,
                            interactive=False
                        )
                        
                        process_details = gr.Textbox(
                            label="Detailed Results",
                            lines=8,
                            interactive=False
                        )
                        
                        clear_status = gr.Textbox(
                            label="Clear Status",
                            lines=1,
                            interactive=False
                        )
            
            # Analytics Tab
            with gr.TabItem("📊 Analytics & Performance"):
                with gr.Row():
                    # Controls
                    with gr.Column(scale=1):
                        gr.HTML("<h3>🎛️ Analytics Controls</h3>")
                        
                        refresh_analytics_btn = gr.Button(
                            "🔄 Refresh Analytics",
                            variant="primary"
                        )
                        
                        benchmark_vs = gr.Dropdown(
                            choices=["chroma", "faiss"],
                            value="chroma",
                            label="Benchmark Vector Store"
                        )
                        
                        run_benchmark_btn = gr.Button(
                            "🏃‍♂️ Run Model Benchmark",
                            variant="secondary"
                        )
                        
                        benchmark_results = gr.Textbox(
                            label="Benchmark Results",
                            lines=10,
                            interactive=False
                        )
                    
                    # Charts
                    with gr.Column(scale=2):
                        gr.HTML("<h3>📈 Performance Charts</h3>")
                        
                        analytics_status = gr.Textbox(
                            label="Analytics Status",
                            lines=1,
                            interactive=False
                        )
                        
                        timeline_plot = gr.Plot(label="Response Time Timeline")
                        model_comparison_plot = gr.Plot(label="Model Performance Comparison")
                        
                        performance_summary = gr.JSON(label="Performance Summary")
        
        # Quick question presets
        quick_questions = [
            "Provide a comprehensive summary of all documents, highlighting the main themes and conclusions.",
            "What methodologies, approaches, or research methods are discussed across the documents?",
            "What are the most significant insights, findings, or recommendations presented?",
            "Extract and summarize all quantitative data, statistics, or numerical findings mentioned."
        ]
        
        # Event handlers
        def chat_handler(question, history, model, vs, compression):
            return advanced_rag.ask_question_advanced(question, history, model, vs, compression)
        
        def process_handler(files, strategy):
            status, details, status_dict = advanced_rag.process_documents_advanced(files, strategy)
            return status, details, status_dict
        
        def clear_handler():
            clear_msg, status_dict = advanced_rag.clear_database()
            return clear_msg, status_dict, []  # Also clear chat
        
        def analytics_handler():
            result, message = advanced_rag.get_performance_analytics()
            if result:
                timeline_fig, model_fig, summary = result
                return message, timeline_fig, model_fig, summary
            else:
                return message, None, None, {}
        
        def benchmark_handler(vs_type):
            return advanced_rag.run_benchmark(vs_type)
        
        def set_quick_question(idx):
            return quick_questions[idx]
        
        # Connect events
        ask_btn.click(
            chat_handler,
            inputs=[question_input, chatbot, model_dropdown, vectorstore_dropdown, compression_checkbox],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(
            chat_handler,
            inputs=[question_input, chatbot, model_dropdown, vectorstore_dropdown, compression_checkbox],
            outputs=[chatbot, question_input]
        )
        
        process_btn.click(
            process_handler,
            inputs=[file_upload, chunk_strategy],
            outputs=[process_status, process_details, status_json]
        )
        
        clear_btn.click(
            clear_handler,
            outputs=[clear_status, status_json, chatbot]
        )
        
        refresh_analytics_btn.click(
            analytics_handler,
            outputs=[analytics_status, timeline_plot, model_comparison_plot, performance_summary]
        )
        
        run_benchmark_btn.click(
            benchmark_handler,
            inputs=[benchmark_vs],
            outputs=[benchmark_results]
        )
        
        # Connect quick question buttons
        for i, btn in enumerate(quick_btns):
            btn.click(
                lambda idx=i: quick_questions[idx],
                outputs=[question_input]
            )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb;">
            <p style="color: #6b7280; font-size: 1.1em;">
                🚀 <strong>Advanced Ultra-Fast RAG Assistant</strong> • 
                Powered by Groq API, LangChain & Gradio • 
                Real-time Analytics & Multi-Model Support
            </p>
        </div>
        """)
    
    return demo

def main():
    """Launch the advanced Gradio interface"""
    demo = create_advanced_gradio_interface()
    
    # Launch with advanced options
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from basic version
        share=False,
        debug=True,
        show_error=True,
        favicon_path=None,
        ssl_verify=False,
        inbrowser=True  # Auto-open browser
    )

if __name__ == "__main__":
    main()
