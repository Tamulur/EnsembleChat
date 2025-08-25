import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="You have not specified a value for the `type` parameter.*")

# Ensure src/ is on sys.path, so `import ensemble_chat` works when running app.py directly
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ensemble_chat.ui.ui import build_ui


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    # Suppress noisy runtime warnings caused by Gradio closing async generators
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*aclose.*was never awaited.*")
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()


