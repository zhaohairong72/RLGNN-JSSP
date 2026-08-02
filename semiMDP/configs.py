"""Global constants shared across the JSSP (Job-Shop Scheduling Problem) simulator.

These constants encode the node / edge / direction "signatures" used to build
and inspect the disjunctive-graph representation of a schedule, plus a few
text tokens used when parsing benchmark instance files.
"""


# ---------------------------------------------------------------------------
# Simulator node types
# Lifecycle state of an operation, also stored as the node attribute ``type``.
# ---------------------------------------------------------------------------
NOT_START_NODE_SIG = -1   # operation has not been scheduled yet
PROCESSING_NODE_SIG = 0  # operation is currently being processed on a machine
DONE_NODE_SIG = 1        # operation has finished processing
DELAYED_NODE_SIG = 2     # operation reserved ahead of time, waiting for its predecessor
DUMMY_NODE_SIG = 3       # synthetic start/end marker nodes (no real work)

# ---------------------------------------------------------------------------
# Simulator edge types
# ---------------------------------------------------------------------------
CONJUNCTIVE_TYPE = 0  # precedence edge *within* a job (constrains op order)
DISJUNCTIVE_TYPE = 1  # "conflict" edge between ops of different jobs sharing a machine

# ---------------------------------------------------------------------------
# Simulator edge directions (applies to conjunctive edges)
# ---------------------------------------------------------------------------
FORWARD = 0   # edge flows earlier-op -> later-op along the job
BACKWARD = 1  # edge flows later-op -> earlier-op (reverse precedence)

# ---------------------------------------------------------------------------
# miscellaneous / text tokens for benchmark file parsing
# ---------------------------------------------------------------------------
N_SEP = 1   # granularity used to round sampled problem sizes
SEP = ' '   # field separator in TA-format benchmark files
NEW = '\n'  # newline marker stripped from the last field of TA files
