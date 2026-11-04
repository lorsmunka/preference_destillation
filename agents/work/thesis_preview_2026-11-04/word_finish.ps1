param([string]$Source, [string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($Source, $false, $false)
    foreach ($toc in $document.TablesOfContents) { $toc.Update() }
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $characters = $document.ComputeStatistics(5)
    $document.SaveAs2($Docx, 16)
    $document.ExportAsFixedFormat($Pdf, 17)
    $document.Close($false)
    Write-Output "pages=$pages characters_with_spaces=$characters"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
