# Check that LiteLLM reaches your ChatGPT plan: one tiny "pong" request per GPT model,
# in Anthropic format (what Claude Code sends). Start deploy\litellm\start.ps1 first.
$tc = Join-Path $env:USERPROFILE ".tokencut"
$key = (Get-Content (Join-Path $tc "litellm_master_key") -Raw).Trim()
$headers = @{ "Authorization" = "Bearer $key"; "anthropic-version" = "2023-06-01" }
foreach ($model in @("gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra")) {
    $body = @{
        model = $model
        max_tokens = 50
        messages = @(@{ role = "user"; content = "Reply with exactly: pong" })
    } | ConvertTo-Json -Depth 5
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:4000/v1/messages" -Method Post `
            -Headers $headers -ContentType "application/json" -Body $body -TimeoutSec 120
        $text = ($r.content | Where-Object { $_.type -eq "text" } | ForEach-Object { $_.text }) -join ""
        Write-Host ("OK    {0,-12} -> {1}" -f $model, $text.Trim())
    } catch {
        $msg = $_.ErrorDetails.Message
        if (-not $msg) { $msg = $_.Exception.Message }
        if ($msg.Length -gt 300) { $msg = $msg.Substring(0, 300) }
        Write-Host ("FAIL  {0,-12} -> {1}" -f $model, $msg)
    }
}
