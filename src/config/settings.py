import os

# ============================================================================
# Dataset Configuration
# ============================================================================

# Choose the knowledge graph dataset: "dbpedia" or "wikidata"
DATASET = os.getenv("DATASET", "wikipedia").lower()

# Dataset-specific endpoints and URI patterns
if DATASET == "wikidata":
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    RESOURCE_URI_PREFIX = "http://www.wikidata.org/entity/"
    PROPERTY_URI_PREFIX = "http://www.wikidata.org/prop/direct/"
    ONTOLOGY_URI_PREFIX = "http://www.wikidata.org/prop/direct/"  # Wikidata uses same prefix for properties
else:  # Default to DBpedia
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    RESOURCE_URI_PREFIX = "http://dbpedia.org/resource/"
    PROPERTY_URI_PREFIX = "http://dbpedia.org/property/"
    ONTOLOGY_URI_PREFIX = "http://dbpedia.org/ontology/"

# ============================================================================
# LLM Configuration
# ============================================================================

# LLM Provider: "gemini", "openai", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Provider-specific configurations
if LLM_PROVIDER == "openai":
    # OpenAI Configuration
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", "YOUR KEY"))
    LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    LLM_BASE_URL = "https://api.openai.com/v1"
    LLM_API_URL = f"{LLM_BASE_URL}/chat/completions"
elif LLM_PROVIDER == "anthropic":
    # Anthropic (Claude) Configuration
    LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", os.getenv("API_KEY", "YOUR KEY"))
    LLM_MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")  # Options: claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229
    LLM_BASE_URL = "https://api.anthropic.com/v1"
    LLM_API_URL = f"{LLM_BASE_URL}/messages"
else:
    # Gemini Configuration (default)
    LLM_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("API_KEY", "YOUR KEY"))
    LLM_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # Options: gemini-2.5-flash-lite, gemini-pro, gemini-1.5-pro
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    LLM_API_URL = f"{LLM_BASE_URL}/models/{LLM_MODEL_NAME}:generateContent"

# Backward compatibility aliases
API_KEY = LLM_API_KEY
MODEL_NAME = LLM_MODEL_NAME
BASE_URL = LLM_BASE_URL
API_URL = LLM_API_URL

# ============================================================================
# Scoring & Ranking Configuration
# ============================================================================

SCORING_IMPORTANCE_WEIGHT = 0.6  # Must sum to 1.0 with similarity weight
SCORING_SIMILARITY_WEIGHT = 0.4

RANKING_PER_ENTITY_LIMIT = 45  # Max triples per entity (k)
RANKING_MIN_SCORE = 0.1  # Minimum final score threshold
RANKING_MAX_TOTAL = 350  # Max total triples in subgraph

SCORING_TRIPLES_PER_ENTITY_LIMIT = 1000  # Max triples per entity during retrieval

SCORING_IMPORTANCE_POWER = 2  # Power function: weight^power
SCORING_IMPORTANCE_MAX_WEIGHT = 5  # Max importance from LLM (1-5 scale)
SCORING_IMPORTANCE_MAX_VALUE = 25  # Max after power: 5^2 = 25
SCORING_TRIPLE_IMPORTANCE_MAX_SUM = 3.0  # Max sum: 3 entities * 1.0

# ============================================================================
# Expansion Configuration
# ============================================================================

EXPANSION_MAX_RESOURCES = 50  # Max resources to expand
EXPANSION_MAX_BATCHES = 10
EXPANSION_BATCH_SIZE = 5  # Resources per batch
EXPANSION_BATCH_LIMIT = 200  # Max triples per batch query

EXPANSION_PRIORITY_FREQUENCY_MULTIPLIER = 10  # For URI frequency in priority
EXPANSION_PRIORITY_IMPORTANCE_MULTIPLIER = 200  # For connected entity importance
EXPANSION_TOP_RESOURCES_DISPLAY = 10  # Top resources to show in logs

# ============================================================================
# Visualization Configuration
# ============================================================================

VIS_MAX_NODES = 100  # Limit to avoid cluttering
VIS_FIGSIZE = (20, 15)  # Width, height in inches
VIS_DPI = 150

VIS_NODE_SIZE_MIN = 300
VIS_NODE_SIZE_MULTIPLIER = 1700  # Node size = min + importance * multiplier

VIS_HIGH_IMPORTANCE_THRESHOLD = 0.7
VIS_MEDIUM_IMPORTANCE_THRESHOLD = 0.4

VIS_COLOR_HIGH_IMPORTANCE = '#FF6B6B'  # Red
VIS_COLOR_MEDIUM_IMPORTANCE = '#4A90E2'  # Blue
VIS_COLOR_LOW_IMPORTANCE = '#95A5A6'  # Gray

VIS_EDGE_COLOR = 'gray'
VIS_EDGE_ALPHA = 0.5
VIS_EDGE_WIDTH = 1.5
VIS_ARROW_SIZE = 20

VIS_LABEL_TRUNCATE_LENGTH = 30  # Truncate longer labels
VIS_MAX_EDGE_LABELS = 50
VIS_MAX_EDGE_LABEL_LENGTH = 20
VIS_FONT_SIZE = 8
VIS_EDGE_FONT_SIZE = 6
VIS_LEGEND_FONT_SIZE = 10
VIS_TITLE_FONT_SIZE = 14

VIS_SPRING_LAYOUT_K = 2  # Optimal node distance
VIS_SPRING_LAYOUT_ITERATIONS = 50
VIS_SPRING_LAYOUT_SEED = 42  # For reproducible layouts

VIS_NODE_ALPHA = 0.8
VIS_EDGE_LABEL_ALPHA = 0.7

VIS_NODE_FILTER_IMPORTANCE_MULTIPLIER = 10
VIS_NODE_FILTER_DEGREE_WEIGHT = 1

# ============================================================================
# Knowledge Graph/SPARQL Configuration
# ============================================================================

KG_MAX_RETRIES = 2
KG_INITIAL_TIMEOUT = 45  # seconds
KG_MAX_TIMEOUT = 90

KG_MAX_UNION_CLAUSES = 100  # Max UNION clauses to avoid 405/500 errors
KG_DEFAULT_TRIPLE_LIMIT = 1000

KG_SPLIT_RESOURCE_THRESHOLD = 10  # Split if more than this many resources
KG_SPLIT_RESOURCE_SMALL_THRESHOLD = 5  # Split if more than this with predicates
KG_SPLIT_PREDICATE_THRESHOLD = 3
KG_BATCH_RESOURCE_SIZE = 5  # Resources per batch when splitting
KG_BATCH_PREDICATE_SIZE = 3  # Predicates per batch when splitting

# Backward compatibility aliases (deprecated, use KG_* versions)
DBPEDIA_MAX_RETRIES = KG_MAX_RETRIES
DBPEDIA_INITIAL_TIMEOUT = KG_INITIAL_TIMEOUT
DBPEDIA_MAX_TIMEOUT = KG_MAX_TIMEOUT
DBPEDIA_MAX_UNION_CLAUSES = KG_MAX_UNION_CLAUSES
DBPEDIA_DEFAULT_TRIPLE_LIMIT = KG_DEFAULT_TRIPLE_LIMIT
DBPEDIA_SPLIT_RESOURCE_THRESHOLD = KG_SPLIT_RESOURCE_THRESHOLD
DBPEDIA_SPLIT_RESOURCE_SMALL_THRESHOLD = KG_SPLIT_RESOURCE_SMALL_THRESHOLD
DBPEDIA_SPLIT_PREDICATE_THRESHOLD = KG_SPLIT_PREDICATE_THRESHOLD
DBPEDIA_BATCH_RESOURCE_SIZE = KG_BATCH_RESOURCE_SIZE
DBPEDIA_BATCH_PREDICATE_SIZE = KG_BATCH_PREDICATE_SIZE

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_BASE_DIR = "out"
OUTPUT_LOG_FILENAME = "log_q{id}.txt"
OUTPUT_RESULTS_FILENAME = "q{id}.txt"
OUTPUT_GRAPH_FILENAME = "graph_q{id}.png"
