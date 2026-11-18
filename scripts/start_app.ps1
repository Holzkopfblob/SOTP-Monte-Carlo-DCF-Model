# Startet die SOTP-App mit dem echten Depot (http://localhost:8501). Fenster offen lassen; Schließen beendet die App.
# Aufruf: Rechtsklick > Mit PowerShell ausführen, oder:  powershell -File scripts\start_app.ps1
$repo = Split-Path -Parent $PSScriptRoot
$env:SOTP_DEPOT = "C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen\depot.json"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $repo
$Host.UI.RawUI.WindowTitle = "SOTP-App (localhost:8501) - Fenster schliessen beendet die App"
& "$repo\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
