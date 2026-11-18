# Monatlicher Labor-Lauf für die Windows-Aufgabenplanung (Aufgabe „SOTP Labor monatlich“).
# Protokoll: <SOTP_DATA>\labor_monat.log (Standard 05_PRIVAT\Finanzen\sotp-daten).
$repo = Split-Path -Parent $PSScriptRoot
$daten = if ($env:SOTP_DATA) { $env:SOTP_DATA } else { "C:\Users\westphal\Documents\Holzkopf\05_PRIVAT\Finanzen\sotp-daten" }
$log = Join-Path $daten "labor_monat.log"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # sonst liest PS 5.1 die Python-Ausgabe als OEM-Codepage (Umlaute kaputt)
Set-Location $repo
"===== $(Get-Date -Format 'yyyy-MM-dd HH:mm') =====" | Out-File -FilePath $log -Append -Encoding utf8
& "$repo\.venv\Scripts\python.exe" -u "$repo\scripts\labor_monat.py" *>&1 |
    Where-Object { $_ -notmatch 'Warning' } | Out-File -FilePath $log -Append -Encoding utf8
"Exit $LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
