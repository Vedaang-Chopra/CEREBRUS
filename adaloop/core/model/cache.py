"""Process-wide holder of the loaded model.

Kept apart from load.py so that `%autoreload 2` reloading load.py does not drop the model.
"""
LOADED: dict = {}  # at most one entry: key -> LoadedVLM
