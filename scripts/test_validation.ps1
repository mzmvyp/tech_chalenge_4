# Script PowerShell para testar validação de predições

# Validar predições por data
$body = @{
    target_date = "2025-11-27"
    actual_price = 4650.0
    use_latest = $true
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "http://localhost:8000/validate-by-date" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

Write-Host "Status: $($response.StatusCode)"
Write-Host "Response:"
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

