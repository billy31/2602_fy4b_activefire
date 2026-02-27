#!/bin/bash
cd /root/codes/fire0226/selfCodes

if [ -n "$(git status --porcelain)" ]; then
    echo "Syncing to GitHub..."
    git add -A
    read -p "Commit message: " msg
    [ -z "$msg" ] && msg="Update code"
    git commit -m "$msg"
    git push origin main && echo "✅ Synced!" || echo "❌ Push failed"
else
    echo "No changes to sync"
fi
