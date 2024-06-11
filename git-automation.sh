#!/bin/bash

# Script to automate git commit and push process

# Function to check if Git is installed
check_git_installed() {
    if ! git --version > /dev/null 2>&1; then
        echo "Git is not installed. Please install Git and try again."
        exit 1
    fi
}

# Check for Git installation
check_git_installed

# Display the status of the repository
echo "Checking the status of your repository..."
git status

# Add changes to the staging area
echo "Adding changes to the staging area..."
git add -A

# Show changes added to the staging area
#echo "Changes added to the staging area:"
#git diff --staged

# Prompt the user for a commit message
read -p "Enter your commit message: " commit_message

# Commit the changes with the provided message
echo "Committing the changes..."
git commit -m "$commit_message"

# Check if the user wants to push the changes
read -p "Do you want to push the changes? (y/n): " push_confirm

if [ "$push_confirm" = "y" ]; then
    # Pull the latest changes from the remote repository
    echo "Pulling the latest changes from the remote repository..."
    git pull

    # Prompt for the branch name to push the changes
    read -p "Enter the branch name to push the changes: " branch_name

    # Push the changes
    echo "Pushing the changes to the branch: $branch_name"
    git push origin "$branch_name"
else
    echo "Changes committed locally. Not pushed to remote repository."
fi

echo "Git operation completed
