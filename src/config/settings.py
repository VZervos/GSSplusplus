import os

# ============================================================================
# API Configuration
# ============================================================================

# Your API key
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR KEY")

# API endpoint for Gemini 2.5 Flash change model name to use different models
MODEL_NAME = "gemini-2.5-flash-lite"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL = f"{BASE_URL}/models/{MODEL_NAME}:generateContent"

# ============================================================================
# Scoring & Ranking Configuration
# ============================================================================

# Final score weights (must sum to 1.0)
SCORING_IMPORTANCE_WEIGHT = 0.6
SCORING_SIMILARITY_WEIGHT = 0.4

# Subgraph selection parameters
RANKING_PER_ENTITY_LIMIT = 45  # How many triples each entity can contribute (k)
RANKING_MIN_SCORE = 0.1  # Minimum final score to include a triple
RANKING_MAX_TOTAL = 350  # Maximum total number of triples in subgraph

# Triple retrieval limits
SCORING_TRIPLES_PER_ENTITY_LIMIT = 1000  # Max triples per entity during initial retrieval

# Importance score computation
SCORING_IMPORTANCE_POWER = 2  # Power function: importance_weight^power
SCORING_IMPORTANCE_MAX_WEIGHT = 5  # Maximum importance weight from LLM (1-5 scale)
SCORING_IMPORTANCE_MAX_VALUE = 25  # Max value after power function (5^2 = 25)
SCORING_TRIPLE_IMPORTANCE_MAX_SUM = 3.0  # Max sum of entity scores in a triple (3 entities * 1.0)

# ============================================================================
# Expansion Configuration
# ============================================================================

# 1-hop expansion parameters
EXPANSION_MAX_RESOURCES = 50  # Maximum number of resources to expand
EXPANSION_MAX_BATCHES = 10  # Maximum number of expansion batches
EXPANSION_BATCH_SIZE = 5  # Number of resources per expansion batch
EXPANSION_BATCH_LIMIT = 200  # Maximum triples per expansion batch query

# Expansion priority computation
EXPANSION_PRIORITY_FREQUENCY_MULTIPLIER = 10  # Multiplier for URI frequency in priority
EXPANSION_PRIORITY_IMPORTANCE_MULTIPLIER = 200  # Multiplier for connected entity importance
EXPANSION_TOP_RESOURCES_DISPLAY = 10  # Number of top resources to display in logs

# ============================================================================
# Visualization Configuration
# ============================================================================

# Graph display limits
VIS_MAX_NODES = 100  # Maximum nodes to display (to avoid cluttering)
VIS_FIGSIZE = (20, 15)  # Figure size in inches (width, height)
VIS_DPI = 150  # Image resolution (dots per inch)

# Node sizing
VIS_NODE_SIZE_MIN = 300  # Minimum node size
VIS_NODE_SIZE_MULTIPLIER = 1700  # Multiplier for importance-based sizing
# Node size formula: VIS_NODE_SIZE_MIN + importance * VIS_NODE_SIZE_MULTIPLIER
# Range: 300 (importance=0) to 2000 (importance=1.0)

# Node coloring thresholds
VIS_HIGH_IMPORTANCE_THRESHOLD = 0.7  # Threshold for high importance
VIS_MEDIUM_IMPORTANCE_THRESHOLD = 0.4  # Threshold for medium importance
# Low importance: score <= VIS_MEDIUM_IMPORTANCE_THRESHOLD

# Node colors (hex codes)
VIS_COLOR_HIGH_IMPORTANCE = '#FF6B6B'  # Red
VIS_COLOR_MEDIUM_IMPORTANCE = '#4A90E2'  # Blue
VIS_COLOR_LOW_IMPORTANCE = '#95A5A6'  # Gray

# Edge styling
VIS_EDGE_COLOR = 'gray'
VIS_EDGE_ALPHA = 0.5
VIS_EDGE_WIDTH = 1.5
VIS_ARROW_SIZE = 20

# Label settings
VIS_LABEL_TRUNCATE_LENGTH = 30  # Truncate node labels longer than this
VIS_MAX_EDGE_LABELS = 50  # Maximum number of edge labels to display
VIS_MAX_EDGE_LABEL_LENGTH = 20  # Maximum length of edge label text
VIS_FONT_SIZE = 8  # Node label font size
VIS_EDGE_FONT_SIZE = 6  # Edge label font size
VIS_LEGEND_FONT_SIZE = 10  # Legend font size
VIS_TITLE_FONT_SIZE = 14  # Title font size

# Layout settings
VIS_SPRING_LAYOUT_K = 2  # Optimal distance between nodes
VIS_SPRING_LAYOUT_ITERATIONS = 50  # Number of iterations for spring layout
VIS_SPRING_LAYOUT_SEED = 42  # Random seed for reproducible layouts

# Transparency
VIS_NODE_ALPHA = 0.8
VIS_EDGE_LABEL_ALPHA = 0.7

# Node filtering for large graphs
VIS_NODE_FILTER_IMPORTANCE_MULTIPLIER = 10  # Multiplier for importance in node filtering score
VIS_NODE_FILTER_DEGREE_WEIGHT = 1  # Weight for degree in node filtering score

# ============================================================================
# DBpedia/SPARQL Configuration
# ============================================================================

# Query retry settings
DBPEDIA_MAX_RETRIES = 2  # Maximum retry attempts for SPARQL queries
DBPEDIA_INITIAL_TIMEOUT = 45  # Initial timeout in seconds
DBPEDIA_MAX_TIMEOUT = 90  # Maximum timeout in seconds

# Query size limits
DBPEDIA_MAX_UNION_CLAUSES = 100  # Maximum UNION clauses in batch queries
DBPEDIA_DEFAULT_TRIPLE_LIMIT = 1000  # Default limit for triple retrieval queries

# Batch query splitting thresholds
DBPEDIA_SPLIT_RESOURCE_THRESHOLD = 10  # Split if more than this many resources
DBPEDIA_SPLIT_RESOURCE_SMALL_THRESHOLD = 5  # Split if more than this with predicates
DBPEDIA_SPLIT_PREDICATE_THRESHOLD = 3  # Split if more than this many predicates
DBPEDIA_BATCH_RESOURCE_SIZE = 5  # Resources per batch when splitting
DBPEDIA_BATCH_PREDICATE_SIZE = 3  # Predicates per batch when splitting

# ============================================================================
# Output Configuration
# ============================================================================

# Output directory structure
OUTPUT_BASE_DIR = "out"  # Base output directory (relative to project root)
OUTPUT_LOG_FILENAME = "log_q{id}.txt"  # Log filename template
OUTPUT_RESULTS_FILENAME = "q{id}.txt"  # Results filename template
OUTPUT_GRAPH_FILENAME = "graph_q{id}.png"  # Graph visualization filename template
