import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="You have not specified a value for the `type` parameter.*")

from ensemble_chat.ui.ui import build_ui
from ensemble_chat.core.console import install_print_colorizer


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        install_print_colorizer()
    except Exception:
        pass
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()


