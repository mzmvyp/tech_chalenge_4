# Script para configurar PATH do Python no PowerShell
# Execute este script uma vez para configurar o PATH permanentemente

Write-Host "Configurando PATH do Python..." -ForegroundColor Green

# Atualizar PATH do usuário
$pythonPath = "C:\Users\Willian\AppData\Local\Programs\Python\Python312"
$pythonScriptsPath = "$pythonPath\Scripts"

# Verificar se já está no PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$pythonPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$pythonPath;$pythonScriptsPath", "User")
    Write-Host "✓ PATH atualizado com sucesso!" -ForegroundColor Green
} else {
    Write-Host "✓ PATH já está configurado." -ForegroundColor Yellow
}

# Atualizar PATH da sessão atual
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host ""
Write-Host "Para aplicar permanentemente, feche e reabra o PowerShell." -ForegroundColor Cyan
Write-Host "Ou execute: refreshenv (se tiver Chocolatey)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Testando Python..." -ForegroundColor Green
python --version

