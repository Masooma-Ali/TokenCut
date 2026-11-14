# One-time ChatGPT sign-in for LiteLLM's chatgpt/ provider (device code).
# Prints a link and a code: open the link, sign in to YOUR ChatGPT account, enter the code.
# If ChatGPT says device-code login is blocked, enable it in ChatGPT Settings > Security.
# The token is saved to ~/.config/litellm/chatgpt/auth.json (private, never commit it).
$py = Join-Path $env:USERPROFILE ".tokencut\litellm-venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
& $py -c "from litellm.llms.chatgpt.authenticator import Authenticator; a = Authenticator(); a.get_access_token(); print('Login OK:', a.auth_file)"
