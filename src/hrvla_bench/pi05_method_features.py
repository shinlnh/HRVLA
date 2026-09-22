"""PI0.5 architecture feature contracts for the shared language router."""

PI05_METHOD_FEATURES = {
    "pi05_st": {"subtask": True, "recovery": False, "retrained": False},
    "pi05_str": {"subtask": True, "recovery": True, "retrained": False},
    "pi05_str_rt": {"subtask": True, "recovery": True, "retrained": True},
}
