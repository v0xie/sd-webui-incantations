import launch

packages = ["sentence_transformers"]
if not launch.is_installed("sentence_transformers"):
    launch.run_pip("install sentence_transformers", "sentence-transformers for Incantation")