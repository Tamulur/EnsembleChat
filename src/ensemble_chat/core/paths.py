from pathlib import Path


def project_root() -> Path:
    """Return the repository root where Configurations/, Prompts/, Settings.json live.

    This walks up from the current file until it finds a directory that contains
    any of these anchors. Falls back to the parent of the src directory.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (
            (parent / "Prompts").exists()
            or (parent / "Configurations").exists()
            or (parent / "Settings.json").exists()
        ):
            return parent
    # Fallback: try to use the directory containing "src" as the project root
    for parent in here.parents:
        if (parent / "src").exists():
            return parent
    # Last resort: go up 3 directories (core -> ensemble_chat -> src -> root)
    try:
        return here.parents[3]
    except IndexError:
        return here.parent


__all__ = ["project_root"]


