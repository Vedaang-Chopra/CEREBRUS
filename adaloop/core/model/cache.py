"""Process-wide holder of the loaded model copies.

Kept apart from load.py so that `%autoreload 2` reloading load.py does not drop the models.
"""
LOADED: dict = {}  # key = (load settings..., copy index) -> LoadedVLM
