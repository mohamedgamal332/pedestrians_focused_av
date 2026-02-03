# GitHub Deployment Guide

## Quick Deployment Steps

This guide walks you through pushing your code to the GitHub repository:
https://github.com/mohamedgamal332/pedestrians_focused_av

### Step 1: Verify Your Git Installation

\\\ash
git --version
\\\

If Git is not installed, download it from: https://git-scm.com/

### Step 2: Configure Git (First Time Only)

\\\ash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
\\\

### Step 3: Navigate to Project Root

\\\ash
cd C:\Users\samso\Downloads\RTMPoseFinetuning\project_root
# Or on Unix/Linux:
# cd ~/Downloads/RTMPoseFinetuning/project_root
\\\

### Step 4: Initialize Git Repository

\\\ash
git init
\\\

### Step 5: Add All Files

\\\ash
git add .
\\\

### Step 6: Create Initial Commit

\\\ash
git commit -m "Initial commit: Pedestrian-focused AV training framework

- GCN model training and evaluation scripts
- RTMPose pose estimation integration
- Comprehensive data loading and preprocessing
- Risk scoring framework for pedestrian behavior
- Visualization and plotting utilities
- Automated training scripts for Windows and Unix"
\\\

### Step 7: Add Remote Repository

\\\ash
git remote add origin https://github.com/mohamedgamal332/pedestrians_focused_av.git
\\\

### Step 8: Verify Remote

\\\ash
git remote -v
\\\

You should see:
\\\
origin  https://github.com/mohamedgamal332/pedestrians_focused_av.git (fetch)
origin  https://github.com/mohamedgamal332/pedestrians_focused_av.git (push)
\\\

### Step 9: Set Default Branch to Main

\\\ash
git branch -M main
\\\

### Step 10: Push to GitHub

\\\ash
git push -u origin main
\\\

**Note:** You may be prompted to enter your GitHub credentials. Use your personal access token as the password.

## After Initial Push

### Checking Status

\\\ash
git status
\\\

### Making Updates

\\\ash
# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
\\\

### Creating Branches for Features

\\\ash
# Create new branch
git checkout -b feature/new-feature-name

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push branch
git push -u origin feature/new-feature-name

# Create Pull Request on GitHub website
\\\

## Complete One-Command Push

If you want to do everything in one go:

\\\ash
cd C:\Users\samso\Downloads\RTMPoseFinetuning\project_root && 
git init && 
git add . && 
git commit -m "Initial commit: Pedestrian-focused AV training framework" && 
git remote add origin https://github.com/mohamedgamal332/pedestrians_focused_av.git && 
git branch -M main && 
git push -u origin main
\\\

## Troubleshooting

### Remote Already Exists

If you get "fatal: remote origin already exists":

\\\ash
git remote remove origin
git remote add origin https://github.com/mohamedgamal332/pedestrians_focused_av.git
\\\

### Authentication Error

For HTTPS with personal access token:

\\\ash
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/mohamedgamal332/pedestrians_focused_av.git
\\\

### SSH Alternative

To use SSH keys instead:

\\\ash
# Generate SSH key (if not already done)
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# Add public key to GitHub (https://github.com/settings/keys)

# Set remote to SSH
git remote set-url origin git@github.com:mohamedgamal332/pedestrians_focused_av.git

# Test connection
ssh -T git@github.com
\\\

## Verifying Your Push

1. Go to: https://github.com/mohamedgamal332/pedestrians_focused_av
2. Verify all files are present
3. Check commit history in GitHub interface

## Next Steps

1. Add collaborators in GitHub Settings
2. Enable GitHub Actions for CI/CD
3. Add branch protection rules
4. Set up GitHub Issues for tracking
5. Create GitHub Wiki for extended documentation

## Additional Commands

### View Commit History

\\\ash
git log --oneline -n 10
\\\

### View Changes

\\\ash
git diff
\\\

### Undo Last Commit (before push)

\\\ash
git reset --soft HEAD~1
\\\

### Undo Last Commit (after push)

\\\ash
git revert HEAD
git push
\\\

## Support

For more Git information: https://git-scm.com/doc
For GitHub help: https://docs.github.com/

