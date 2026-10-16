param([string]$OutputDir = "$PSScriptRoot/../.artifacts/audio")
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
[IO.Directory]::CreateDirectory([IO.Path]::GetFullPath($OutputDir)) | Out-Null
$items = @{
  'initial-en' = 'This battery is different from that one. It goes in the device. The other one is for that, and it is different. I think that is why we should use this.'
  'answer-en' = 'Battery A is the main battery. Battery B is the backup battery. They work in the same device, not as interchangeable alternatives.'
  'retry-en' = 'This device has two batteries. Battery A is the main battery and Battery B is the backup. Their voltage and capacity differ, so they cannot be interchanged. Use the battery specified for each position.'
}
foreach ($name in $items.Keys) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $s.SelectVoice('Microsoft David Desktop')
    $s.SetOutputToWaveFile([IO.Path]::GetFullPath((Join-Path $OutputDir "$name.wav")))
    $s.Speak($items[$name])
  } finally { $s.Dispose() }
}
Write-Output 'Created three synthetic English speech fixtures (not human microphone or Hindi evidence).'
