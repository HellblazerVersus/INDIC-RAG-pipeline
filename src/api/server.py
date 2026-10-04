"""FastAPI Web Server for Voice-Enabled RAG System (Indic MSMARCO-XI)."""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.generation.generator import get_generator
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.safety import CompositeGuardrail
from src.pipeline.harness import RobustExecutionHarness
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.schemas import AudioInputRequest, RAGResponse, TextInputRequest
from src.stt.transcriber import get_transcriber
from src.utils.logging import logger

# Global pipeline instance
pipeline_instance: Optional[RAGPipeline] = None
index_manager_instance: Optional[Any] = None
config_data: Dict[str, Any] = {}


def load_app_config(config_path: str = "configs/config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, index_manager_instance, config_data
    logger.info("Initializing Voice RAG Pipeline for Web Service...")

    config_data = load_app_config()
    retrieval_cfg = config_data.get("retrieval", {})
    stt_cfg = config_data.get("stt", {})
    guard_cfg = config_data.get("guardrails", {})
    gen_cfg = config_data.get("generation", {})

    device = retrieval_cfg.get("device", "cuda")
    embed_model_name = retrieval_cfg.get("embedding_model", "intfloat/multilingual-e5-small")
    index_path = retrieval_cfg.get("index_path", "data/processed/faiss_index.bin")
    metadata_path = retrieval_cfg.get("metadata_path", "data/processed/passage_metadata.json")

    retrieval_mode = retrieval_cfg.get("mode", "dense")
    if retrieval_mode == "bm25":
        logger.info("Initializing BM25 Sparse Retriever (Low Memory Mode)...")
        from src.retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(metadata_path=metadata_path, top_k=retrieval_cfg.get("top_k", 5))
    else:
        # 1. Initialize Embedder
        logger.info(f"Loading embedder: {embed_model_name} on {device}...")
        from src.retrieval.embedder import MultilingualE5Embedder
        embedder = MultilingualE5Embedder(model_name_or_path=embed_model_name, device=device, warmup=True)
    
        # 2. Initialize / Load FAISS Index
        from src.retrieval.index import FAISSIndexManager
        index_manager = FAISSIndexManager(dimension=embedder.dimension, index_type=retrieval_cfg.get("index_type", "FlatIP"))
        if Path(index_path).exists() and Path(metadata_path).exists():
            logger.info(f"Loading FAISS index from {index_path}...")
            index_manager.load(index_path, metadata_path)
        else:
            logger.warning("FAISS index not found on disk. Initializing bootstrap index...")
            from benchmarks.run_latency_bench import ensure_index_exists
            index_manager = ensure_index_exists(embedder, index_path, metadata_path)
    
        index_manager_instance = index_manager
    
        # 3. Retriever
        from src.retrieval.retriever import Retriever as DenseRetriever
        retriever = DenseRetriever(embedder=embedder, index_manager=index_manager, top_k=retrieval_cfg.get("top_k", 5))

    # 4. Guardrail
    min_confidence = guard_cfg.get("min_confidence_threshold", 0.75)
    guardrail = CompositeGuardrail(
        confidence_threshold=min_confidence,
        enable_input_safety=guard_cfg.get("enable_input_safety", True),
        enable_off_topic=guard_cfg.get("enable_off_topic_detection", True),
        enable_groundedness=guard_cfg.get("enable_groundedness_check", True),
    )

    # 5. Generator
    generator = get_generator(
        provider=gen_cfg.get("provider", "mock"),
        model_name=gen_cfg.get("model_name", "mock-llm-indic"),
        temperature=gen_cfg.get("temperature", 0.1),
    )

    # 6. STT Transcriber
    stt_provider = stt_cfg.get("provider", "sarvam")
    logger.info(f"Setting up STT Transcriber ({stt_provider})...")
    transcriber = get_transcriber(
        provider=stt_provider,
        model_size=stt_cfg.get("model_size", "tiny"),
        device=device,
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 1),
        vad_filter=stt_cfg.get("vad_filter", False),
    )

    # 7. Orchestration Harness & Pipeline
    harness = RobustExecutionHarness(
        max_retries=config_data.get("harness", {}).get("max_retries", 3),
        backoff_factor=config_data.get("harness", {}).get("backoff_factor", 1.5),
        enable_circuit_breaker=config_data.get("harness", {}).get("enable_circuit_breaker", True),
    )

    pipeline_instance = RAGPipeline(
        transcriber=transcriber,
        retriever=retriever,
        guardrail=guardrail,
        generator=generator,
        harness=harness,
        default_language=config_data.get("dataset", {}).get("language", "hi"),
    )

    logger.info("Voice RAG Web Pipeline successfully initialized.")
    yield
    logger.info("Shutting down Voice RAG Web Pipeline...")


app = FastAPI(
    title="Indic Voice-Enabled RAG API (Personal Project 2026)",
    description="Sub-200ms Voice & Text Indic Retrieval-Augmented Generation system on MSMARCO-XI.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the interactive web interface."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html


class TextQueryRequestModel(BaseModel):
    query: str = Field(..., description="Query text in Hindi or English")
    language: str = Field("hi", description="ISO language code ('hi' or 'en')")


@app.post("/api/query/text", response_model=RAGResponse)
async def api_query_text(req: TextQueryRequestModel):
    """Processes a text query through Retrieval -> Guardrails -> Generation."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized")
    try:
        response = pipeline_instance.process_text(TextInputRequest(query=req.query, language=req.language))
        return response
    except Exception as e:
        logger.error(f"[API] Error in query_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/voice", response_model=RAGResponse)
async def api_query_voice(file: UploadFile = File(...), language: str = "hi"):
    """Processes a voice audio file through STT -> Retrieval -> Guardrails -> Generation."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized")

    temp_path = None
    try:
        content = await file.read()
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        response = pipeline_instance.process_voice(AudioInputRequest(audio_path=temp_path, language=language))
        return response
    except Exception as e:
        logger.error(f"[API] Error in query_voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline_instance is not None,
        "index_vectors": index_manager_instance.total_vectors if index_manager_instance else 0,
    }


@app.get("/api/stats")
async def get_stats():
    """Returns vector index and system metadata statistics."""
    if not index_manager_instance:
        raise HTTPException(status_code=503, detail="Index manager not initialized")
    return {
        "total_vectors": index_manager_instance.total_vectors,
        "dimension": index_manager_instance.dimension,
        "index_type": index_manager_instance.index_type,
        "config": {
            "dataset": config_data.get("dataset", {}),
            "chunking": config_data.get("chunking", {}),
            "retrieval": config_data.get("retrieval", {}),
            "stt": config_data.get("stt", {}),
        },
    }
