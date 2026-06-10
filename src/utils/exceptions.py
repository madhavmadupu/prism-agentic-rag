class PrismError(Exception):
    """Base exception for all P.R.I.S.M. errors."""


class ConfigurationError(PrismError):
    """Raised when required configuration is missing or invalid."""


class RetrievalError(PrismError):
    """Raised when vector retrieval or reranking fails."""


class EmbeddingError(PrismError):
    """Raised when embedding generation fails."""


class IngestionError(PrismError):
    """Raised when document ingestion into Pinecone fails."""


class GraphQueryError(PrismError):
    """Raised when Neo4j query execution or Cypher generation fails."""


class MCPError(PrismError):
    """Raised when a Model Context Protocol server call fails."""


class MultimodalError(PrismError):
    """Raised when vision model extraction fails."""


class EvaluationError(PrismError):
    """Raised when RAGAS or TruLens evaluation fails."""
