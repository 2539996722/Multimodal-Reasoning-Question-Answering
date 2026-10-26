from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_enterprise_rag_dependencies_are_declared() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()

    for dependency in ("langchain-core", "langchain-text-splitters", "langchain-openai", "numpy", "faiss-cpu"):
        assert dependency in requirements
    assert "langchain-community" not in requirements
    assert "faiss-cpu" in requirements
    assert not any(name in requirements for name in ("chromadb", "milvus", "qdrant", "weaviate", "pinecone"))


def test_prd_covers_enterprise_index_lifecycle_and_homepage_acceptance() -> None:
    prd = (ROOT / "prd.md").read_text(encoding="utf-8")
    required_terms = (
        "LangChain",
        "FAISS",
        "OpenAI-compatible",
        "single-flight",
        "immutable snapshot",
        "原子",
        "LRU",
        "index_build_id",
        "降级顺序",
        "首页总览",
        "1440×900",
        "1024×768",
        "390×844",
    )

    for term in required_terms:
        assert term in prd


def test_reusable_subagent_prompts_define_ownership_delivery_and_acceptance() -> None:
    prompts = (ROOT / "3个subagent的提示词.md").read_text(encoding="utf-8")

    assert prompts.count("文件所有权") >= 3
    assert prompts.count("必须交付") >= 3
    assert prompts.count("自验收") == 3
    assert prompts.count("完成汇报必须包含") == 3
    for term in ("LangChain", "FAISS", "OpenAI-compatible", "无数据库", "原子发布", "降级"):
        assert term in prompts


def test_readme_documents_runtime_cache_security_and_operations() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for term in (
        "LangChain",
        "FAISS",
        "runtime/rag-index",
        "single-flight",
        "immutable ready snapshot",
        "原子发布",
        "可观测性、评测与降级",
        "首页产品总览验收",
        "vectors.npy",
        "documents.json",
        "manifest.json",
        "禁止 `allow_dangerous_deserialization`",
    ):
        assert term in readme
    assert "仅包含 `index.faiss`" not in readme
