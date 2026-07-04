# VESSL Workspace skill

Manage the `freuid-hy` VESSL workspace over SSH.

**SSH alias:** `freuid-hy` (configured in `~/.ssh/config`, key at `~/.ssh/freuid.pem`)
**Repo path on workspace:** `/root/repo`
**Python:** `/opt/conda/bin/python3`
**Training log:** `/tmp/train_baseline_v0.log` (or `/tmp/train_<name>.log` for other configs)

## What to do based on the user's request

### Check training progress
```bash
ssh freuid-hy "grep -av 'it/s\|it]' /tmp/train_<name>.log"
```
Replace `<name>` with the config name (e.g. `baseline_v0`). This strips tqdm progress bar lines and shows only epoch summaries and key events.

To find all active log files:
```bash
ssh freuid-hy "ls /tmp/train_*.log 2>/dev/null"
```

### Check if training is still running
```bash
ssh freuid-hy "ps aux | grep 'freuid.train' | grep -v grep"
```

### Start a training run
Always use `nohup` so it persists after SSH disconnects:
```bash
ssh freuid-hy "cd /root/repo && nohup /opt/conda/bin/python3 -m freuid.train --config configs/<name>.yaml > /tmp/train_<name>.log 2>&1 & echo $!"
```
Then verify it started cleanly (wait ~30s for the sanity check):
```bash
ssh freuid-hy "grep -av 'it/s\|it]' /tmp/train_<name>.log | head -10"
```

### Pull latest code to workspace
First check for uncommitted changes (usually notebook outputs — safe to discard):
```bash
ssh freuid-hy "cd /root/repo && git status --short"
ssh freuid-hy "cd /root/repo && git checkout -- notebooks/ && git pull origin <branch> 2>&1"
```
Never reset `data/` — it is gitignored and not touched by git.

### Run inference / generate submission
```bash
ssh freuid-hy "cd /root/repo && nohup /opt/conda/bin/python3 -m freuid.infer --checkpoint checkpoints/<name>.pt --out submissions/<name>.csv > /tmp/infer_<name>.log 2>&1 & echo $!"
```

### Submit to Kaggle from workspace
```bash
ssh freuid-hy "cd /root/repo && /opt/conda/bin/kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f submissions/<name>.csv -m '<message>'"
```

### Check data is intact
```bash
ssh freuid-hy "ls /root/repo/data/train/train | wc -l && ls /root/repo/data/public_test/public_test | wc -l"
```
Expected: 69352 train, 7821 test.

### Open a shell on the workspace
Tell the user to run in their terminal:
```
ssh freuid-hy
```

## Key facts
- Training runs survive laptop shutdown — started with `nohup`, detached from SSH session.
- Checkpoints save to `/root/repo/checkpoints/<config-name>.pt` on best `probe_audet`.
- Data at `/root/repo/data/` is gitignored — git operations never touch it.
- The workspace uses conda Python, not system Python. Always use `/opt/conda/bin/python3`.
- After pushing a local fix, always `git pull` on the workspace before restarting training.
