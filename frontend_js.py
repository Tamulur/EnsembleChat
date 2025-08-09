"""
Centralized JavaScript snippets used by Gradio UI callbacks.

These are provided as plain strings for the `js=` argument in Gradio events.
"""

# Align the beginning of the newest assistant message to the top of the visible chat area
JS_ALIGN_ON_CHANGE = """
() => {
  const app = document.querySelector('gradio-app');
  const doc = (app && app.shadowRoot) ? app.shadowRoot : document;
  const root = doc.getElementById('chat_interface');
  if (!root) return;
  const findScrollableAncestor = (el) => {
    // Stable selection for the chat scroll container
    const stable = root.querySelector("div.bubble-wrap[role='log'][aria-label='chatbot conversation']");
    if (stable) return stable;
    const preferred = root.querySelector('.bubble-wrap') || root.querySelector('.wrapper');
    if (preferred && preferred.scrollHeight > preferred.clientHeight) return preferred;
    let node = el ? el.parentElement : null;
    while (node && node !== root) {
      const cs = getComputedStyle(node);
      const scrollable = (cs.overflowY === 'auto' || cs.overflowY === 'scroll') && node.scrollHeight > node.clientHeight;
      if (scrollable) return node;
      node = node.parentElement;
    }
    return root;
  };
  const findTarget = () => {
    const candidates = [
      "div.bubble-wrap[role='log'][aria-label='chatbot conversation'] div.message-wrap > div.message-row.bubble.bot-row:last-of-type",
      "div.message-wrap > div.message-row.bubble.bot-row:last-of-type",
      "div.svelte-1csv61q.latest .message-content",
      "div.svelte-1csv61q.latest"
    ];
    for (const sel of candidates) {
      const el = root.querySelector(sel);
      if (el) return el;
    }
    const items = root.querySelectorAll("[role='listitem'] , li, .message");
    return items && items.length ? items[items.length - 1] : null;
  };
  const relativeTop = (target, container) => {
    let y = 0;
    let n = target;
    while (n && n !== container) { y += n.offsetTop || 0; n = n.offsetParent; }
    return y;
  };
  const alignTop = () => {
    const target = findTarget();
    if (!target) return;
    const container = findScrollableAncestor(target);
    const top = relativeTop(target, container);
    if (typeof container.scrollTo === 'function') { container.scrollTo({ top, behavior: 'auto' }); } else { container.scrollTop = top; }
    if (typeof target.scrollIntoView === 'function') { target.scrollIntoView({ block: 'start', inline: 'nearest', behavior: 'auto' }); }
  };
  // Multiple attempts to override built-in autoscroll and layout reflows
  let i = 0;
  const tries = () => {
    alignTop();
    if (i++ < 8) requestAnimationFrame(tries);
  };
  requestAnimationFrame(tries);
  setTimeout(alignTop, 60);
  setTimeout(alignTop, 220);
}
"""

# After server event completes, apply a scroll fix to align the newest assistant message
JS_SCROLL_FIX_AFTER_EVENT = """
() => {
  const app = document.querySelector('gradio-app');
  const doc = (app && app.shadowRoot) ? app.shadowRoot : document;
  const root = doc.getElementById('chat_interface');
  if (!root) return;
  const findScrollableAncestor = (el) => {
    const stable = root.querySelector("div[role='log'][aria-label='chatbot conversation']");
    if (stable) return stable;
    let node = el ? el.parentElement : null;
    while (node && node !== root) {
      const cs = getComputedStyle(node);
      const scrollable = (cs.overflowY === 'auto' || cs.overflowY === 'scroll') && node.scrollHeight > node.clientHeight;
      if (scrollable) return node;
      node = node.parentElement;
    }
    return root;
  };
  const findTarget = () => {
    const candidates = [
      "div.message.svelte-1csv61q.panel-full-width div.svelte-1csv61q.latest div.message-content span.md.svelte-7ddecg.chatbot.prose p",
      "div.svelte-1csv61q.latest .message-content span.md.chatbot.prose p",
      "div.svelte-1csv61q.latest .message-content",
      "div.svelte-1csv61q.latest",
      "div.message-row.bubble.bot-row.svelte-1csv61q:last-of-type",
      ".bot-row:last-of-type"
    ];
    for (const sel of candidates) {
      const el = root.querySelector(sel);
      if (el) return el;
    }
    const items = root.querySelectorAll("[role='listitem'] , li, .message");
    return items && items.length ? items[items.length - 1] : null;
  };
  const alignTop = () => {
    const target = findTarget();
    if (!target) return;
    const container = findScrollableAncestor(target);
    const top = target.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
    container.scrollTop = top;
    if (typeof target.scrollIntoView === 'function') { target.scrollIntoView({ block: 'start', inline: 'nearest', behavior: 'auto' }); }
  };
  let i = 0;
  const tries = () => {
    alignTop();
    if (i++ < 12) requestAnimationFrame(tries);
  };
  requestAnimationFrame(tries);
  setTimeout(alignTop, 60);
  setTimeout(alignTop, 220);
  setTimeout(alignTop, 500);
}
"""

# Preserve per-tab scroll position across tab switches
JS_PRESERVE_TAB_SCROLL = """
() => {
  const app = document.querySelector('gradio-app');
  const doc = (app && app.shadowRoot) ? app.shadowRoot : document;
  const scrollPositions = new Map();
  const ids = ['chat_interface', 'o3_view', 'claude_view', 'gemini_view'];
  const getEntries = () => {
    const out = [];
    for (const id of ids) {
      const host = doc.getElementById(id);
      if (!host) continue;
      const stable = host.querySelector("div.bubble-wrap[role='log'][aria-label='chatbot conversation']");
      const container = stable || host.querySelector('.bubble-wrap') || host.querySelector('.wrapper') || host;
      out.push({ key: id, host, container });
    }
    return out;
  };
  const isVisible = (el) => {
    if (!el) return false;
    const panel = el.closest("[role='tabpanel']") || el;
    return !!(panel && panel.offsetParent !== null);
  };
  const saveVisible = () => {
    for (const { key, host, container } of getEntries()) {
      if (isVisible(host)) { scrollPositions.set(key, container.scrollTop); }
    }
  };
  const restoreVisible = () => {
    for (const { key, host, container } of getEntries()) {
      if (isVisible(host) && scrollPositions.has(key)) { container.scrollTop = scrollPositions.get(key); }
    }
  };
  const onTabClick = (e) => {
    const btn = e.target && e.target.closest && e.target.closest("button[role='tab']");
    if (!btn) return;
    saveVisible();
    requestAnimationFrame(restoreVisible);
    setTimeout(restoreVisible, 60);
    setTimeout(restoreVisible, 200);
  };
  doc.addEventListener('click', onTabClick, true);
  const observer = new MutationObserver(() => { restoreVisible(); });
  const startObserve = () => {
    const panels = doc.querySelectorAll("[role='tabpanel']");
    panels.forEach(p => observer.observe(p, { attributes: true, attributeFilter: ['style', 'class', 'hidden'] }));
  };
  startObserve();
  restoreVisible();
}
"""


