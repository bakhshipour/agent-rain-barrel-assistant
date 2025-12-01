# GitHub Repository Setup Script
# Run this after creating your GitHub repository

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Set Git Config (if not already set)
Write-Host "Step 1: Configure Git (if needed)" -ForegroundColor Yellow
$currentName = git config --global user.name
$currentEmail = git config --global user.email

if (-not $currentName) {
    $name = Read-Host "Enter your name for Git commits"
    git config --global user.name $name
    Write-Host "✓ Git name configured" -ForegroundColor Green
} else {
    Write-Host "✓ Git name already configured: $currentName" -ForegroundColor Green
}

if (-not $currentEmail) {
    $email = Read-Host "Enter your email for Git commits"
    git config --global user.email $email
    Write-Host "✓ Git email configured" -ForegroundColor Green
} else {
    Write-Host "✓ Git email already configured: $currentEmail" -ForegroundColor Green
}

Write-Host ""

# Step 2: Get GitHub repository URL
Write-Host "Step 2: Connect to GitHub Repository" -ForegroundColor Yellow
Write-Host ""
Write-Host "First, create a repository on GitHub:" -ForegroundColor White
Write-Host "  1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "  2. Repository name: rain-barrel-assistant" -ForegroundColor White
Write-Host "  3. Description: AI-powered multi-agent system for rain barrel operations" -ForegroundColor White
Write-Host "  4. Make it PUBLIC (required for Kaggle)" -ForegroundColor White
Write-Host "  5. DO NOT initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "  6. Click 'Create repository'" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/rain-barrel-assistant.git)"

if ($repoUrl) {
    # Remove existing remote if any
    $existingRemote = git remote get-url origin 2>$null
    if ($existingRemote) {
        Write-Host "Removing existing remote..." -ForegroundColor Yellow
        git remote remove origin
    }
    
    # Add new remote
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin $repoUrl
    
    # Rename branch to main (if on master)
    $currentBranch = git branch --show-current
    if ($currentBranch -eq "master") {
        Write-Host "Renaming branch to 'main'..." -ForegroundColor Yellow
        git branch -M main
    }
    
    # Push to GitHub
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "You may be prompted for GitHub credentials." -ForegroundColor White
    Write-Host ""
    
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your repository is now live at:" -ForegroundColor Cyan
        Write-Host $repoUrl -ForegroundColor White
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Verify all files are on GitHub" -ForegroundColor White
        Write-Host "  2. Check that .env files are NOT visible" -ForegroundColor White
        Write-Host "  3. Update KAGGLE_SUBMISSION.md with repository URL" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "Push failed. Common issues:" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "  1. Authentication: Use GitHub Personal Access Token" -ForegroundColor White
        Write-Host "  2. Repository doesn't exist: Create it first on GitHub" -ForegroundColor White
        Write-Host "  3. Wrong URL: Check the repository URL" -ForegroundColor White
        Write-Host ""
        Write-Host "For authentication, you may need to:" -ForegroundColor Yellow
        Write-Host "  - Use a Personal Access Token instead of password" -ForegroundColor White
        Write-Host "  - Or use GitHub CLI: gh auth login" -ForegroundColor White
    }
} else {
    Write-Host "No URL provided. Exiting." -ForegroundColor Red
}

