from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "P.R.I.S.M."
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embeddings & Retrieval
    cohere_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "prism-index"
    pinecone_environment: str = "us-east-1-aws"

    # Graph Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Cache
    redis_url: str = "redis://localhost:6379"

    # MCP
    mcp_server_port: int = 5000
    yahoo_finance_api_key: str = ""


settings = Settings()
