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
    "/* Per-button icon sizing: target the <img> inside each button by elem_id */\n"
    "#btn_chatgpt .icon, #btn_chatgpt img { height: 22px !important; width: auto; }\n"
    "#btn_claude .icon, #btn_claude img { height: 22px !important; width: auto; }\n"
    "#btn_gemini .icon, #btn_gemini img { height: 22px !important; width: auto; }\n"
    "#btn_chatgpt_claude .icon, #btn_chatgpt_claude img { height: 24px !important; width: auto; }\n"
    "#btn_all .icon, #btn_all img { height: 25px !important; width: auto; }\n"
)


