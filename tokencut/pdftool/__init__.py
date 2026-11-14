"""read_pdf: a local PDF reader for the agent (MCP tool), text pages as text.

Runs NEXT TO the agent, like the repo index; the gateway still never reads the disk.
Claude Code with Poppler reads long PDFs as page pictures only, so the gateway's PDF rule
(attachments step 7) never sees a PDF. This tool applies the same page check where the file
is: pages with only text come back as text, pages with charts, tables, images or scans come
back as pictures. A PreToolUse hook (`python -m tokencut.pdftool hook`) sends Claude Code's
Read of a .pdf to this tool, so the choice does not depend on the model following a prompt.
"""
