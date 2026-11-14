@echo off
REM One-time: turn this folder into a git repo and push it to an EMPTY GitHub repository.
REM Needs Git for Windows (includes Git LFS). Create the empty repo on github.com first
REM (no README, no .gitignore), then paste its URL when asked.
setlocal
cd /d "%~dp0"
where git >nul 2>&1 || ( echo Git not found. Install Git for Windows from https://git-scm.com & pause & exit /b 1 )
git lfs version >nul 2>&1 || ( echo Git LFS not found. Install it from https://git-lfs.com & pause & exit /b 1 )

if not exist ".git" (
    git init -b main || ( pause & exit /b 1 )
)
git lfs install --local
git lfs track "*.safetensors" >nul

echo Staging files (model weights go through Git LFS)...
git add -A
git diff --cached --quiet && echo Nothing new to commit.
git diff --cached --quiet || git commit -m "Phyllo-ViT v3: reliability-aware phyllotaxy recognition, API, frontend, experiments"
if errorlevel 1 ( echo Commit failed. If git asks for your name/email run: & echo   git config --global user.name "Your Name" & echo   git config --global user.email "you@example.com" & pause & exit /b 1 )

git remote get-url origin >nul 2>&1 && goto push
set /p REPO_URL=Paste your empty GitHub repo URL (e.g. https://github.com/you/phyllo-vit.git): 
git remote add origin "%REPO_URL%" || ( pause & exit /b 1 )
:push
echo Pushing (first push uploads ~700 MB of model weights via LFS, this can take a while)...
git push -u origin main
if errorlevel 1 ( echo Push failed - check the URL, your GitHub login, and your Git LFS quota. & pause & exit /b 1 )
echo.
echo Done. Your code is on GitHub.
pause
