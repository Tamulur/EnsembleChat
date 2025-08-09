"""
Centralized CSS used by Gradio Blocks.
"""

CSS_GLOBAL = (
    "footer {visibility: hidden}\n"
    "#chat_interface .wrapper,\n"
    "#chat_interface .wrapper.svelte-g3p8na {\n"
    "  scroll-behavior: auto !important;\n"
    "  overscroll-behavior: contain;\n"
    "}\n"
    "#chat_interface .bubble-wrap,\n"
    "#chat_interface .bubble-wrap.svelte-gjtrl6 {\n"
    "  scroll-behavior: auto !important;\n"
    "}\n"
    "#chat_interface .wrapper * { overflow-anchor: none; }\n"
    "/* Hide Chatbot clear/trash icon inside the main chat */\n"
    "#chat_interface button[aria-label*='clear' i],\n"
    "#chat_interface button[title*='clear' i],\n"
    "#chat_interface button[aria-label*='delete' i],\n"
    "#chat_interface button[title*='delete' i] {\n"
    "  display: none !important;\n"
    "}\n"
)


