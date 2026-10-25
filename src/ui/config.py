class UIConfig:
    PORT = 8080
    HOST = "127.0.0.1"
    DATASET_PATH = "../dkdata.json"
    CHECKPOINT_PATH = "artifacts/models/experiment3_graphsage.pt"
    TOP_K = 15
    GRAPH_RADIUS = 1
    MAX_VISIBLE_NEIGHBORS = 15
    DEFAULT_MIN_TX = 1

    # Cycle index paging.  A transport bound, not a detection threshold: it
    # decides how many summary rows travel in one response, never which cycles
    # exist.  Measured on the Reference Dataset (5,589 cycles): a 100-row page
    # is ~47 KB, the whole index would be ~3.1 MB, and the full evidence for
    # every cycle is ~19.3 MB.  Every response states the page in force and
    # whether more remain.
    CYCLE_PAGE_SIZE = 100
    CYCLE_PAGE_MAX_SIZE = 500
