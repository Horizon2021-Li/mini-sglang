"""Capture an HTML view of the actual run.log (requires Playwright Chromium)."""
import html
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent
raw = (root / 'run.log').read_text()
assert 'PASS: real Llama weights loaded' in raw, 'Run must succeed before making screenshots'
# Remove progress control sequences only; retain the source log separately.
log = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', raw).replace('\r', '\n')
first, second = log.split('[3] Manually constructed Req and Batch', 1)
sections = [('01-model-config', '4.1 / 4.2 — Llama configuration and weight loading', first),
            ('02-req-batch', '4.1 / 4.2 — Req, Batch and Context', '[3] Manually constructed Req and Batch' + second)]
body = ''
for key, title, content in sections:
    body += f'<section id="{key}"><h1>{html.escape(title)}</h1><p>Actual script output · TinyLlama 1.1B · Mini-SGLang · GPU FP16</p><pre>{html.escape(content.strip())}</pre><footer>Source: minimal_llama.py → run.log | Log viewer screenshot; no inference executed</footer></section>'
page_html = '''<!doctype html><meta charset="utf-8"><title>Mini-SGLang run evidence</title>
<style>body{margin:0;background:#0b1020;color:#e8effb;font:16px monospace}section{padding:32px;width:1100px;box-sizing:border-box;background:#111a2e;margin:20px}h1{font:24px sans-serif;color:#9ad9ff}p,footer{color:#9caec8;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}footer{border-top:1px solid #344057;padding-top:16px}</style>''' + body
report = root / 'run-report.html'
report.write_text(page_html)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width':1140, 'height':900}, device_scale_factor=1)
    page.goto(report.as_uri())
    for key, _, _ in sections:
        page.locator(f'[id="{key}"]').screenshot(path=str(root / (key + '.png')))
    browser.close()
print('Saved run-report.html and two screenshots of actual logged output.')
