# YODHA runbook: what is running, how to check it, what to do next

## 1. What is running right now
- **Your Mac** is training the agent: `runs/mac_main`, about 500 steps per second.
- It runs in the background and keeps going if you close the terminal or the chat. Don't shut down the Mac or close the lid.

### Check on it (any time)
- **Last 3 lines of the log:**
  ```bash
  cd ~/Hackathon/Agent_Yodha && grep -v Warning runs/mac_main.log | tail -3
  ```
- **Graphs:**
  ```bash
  uv run tensorboard --logdir runs
  ```
  Then open http://localhost:6006.

**What good numbers look like:**
- `agree`: how often the student does the same as the coach. It should rise above 0.85.
- `T1 exit`, `T2 exit`, `T3 exit`: the fraction of maps finished. These should go up over the hours.
- `dead`: should go down.
- `sps`: speed. About 400–550 is normal on the Mac.

**Is it still alive?**
```bash
pgrep -f kickstart_ppo
```
Numbers printed means it's running. Nothing printed means it stopped; tell Claude.

### Autopilot: no need to ask Claude every few hours
A second background program (`training/eval/autopilot.py`) checks everything **every 2 hours** by itself:
- tests the newest model on the same unseen maps,
- records 2 videos,
- **pushes the new model to GitHub if it scores clearly better** (only after `validate.py` passes),
- restarts training if it stopped,
- shows a Mac notification.

| To | Do |
|---|---|
| Read the report | Open `runs/STATUS.md` (plain language: what it can do per level type, score history) |
| Watch the latest videos | `open runs/videos/latest_T1.mp4` (fight) and `open runs/videos/latest_T3.mp4` (key, then locked door, then exit) |
| Watch a fresh video right now | `scripts/preview.sh 3` (any tier 1–4; takes about 1 min, then the video opens) |
| Check the autopilot is alive | `pgrep -f training.eval.autopilot` |
| Stop the autopilot | `pkill -f training.eval.autopilot` |

## 2. Kaggle: a second computer training for free (do this when Claude says "ckpt-A is published")
1. Go to kaggle.com, click your picture, then **Settings**. Verify your **phone number** (one time only).
2. Click **Create → New Notebook**.
3. In the right panel, open **Session options**:
   - **Accelerator = GPU T4 x2**.
   - **Internet = On**.
4. Delete the example cell. Open the file `training/cloud/kaggle_cell.py` from the repo, copy all of it, and paste it into one cell.
5. Click **Save Version** (top right), choose **Save & Run All (Commit)**, then **Save**.
   - It now runs in the background for up to 11 hours, even if you close the browser.
6. **To check on it:** go to kaggle.com → **Your Work** → your notebook → the running version → **Logs**.
   - Good sign: lines like `[12] steps 3.10M phase B ... sps 250` keep appearing.
7. **When it's done** (or after about 10 hours):
   1. Open the notebook → **Output** tab and download the `kaggle_export` folder.
   2. Put it in `~/Hackathon/Agent_Yodha/runs/kaggle_b_export/`.
   3. Tell Claude. Claude compares it with the Mac model and keeps the better one.

## 3. Codespaces: the final "is it exactly like the judges' machine?" check (near the end)
1. Go to github.com/AnshulMohanty/Agent_Yodha → **Code → Codespaces** and open your codespace.
2. In its terminal, run:
   ```bash
   git pull && git lfs pull
   pip install uv && uv sync
   uv run python validate.py --docker
   ```
3. **Good:** the last line says **PASSED**. The CPU-limit and RAM lines show real numbers here, not warnings.
4. Optional: also set **Settings → Codespaces → Default idle timeout = 240 minutes**.

## 4. Timeline
| When | What | Who |
|---|---|---|
| Now → about 2 h | The student copies the coach (phase A, 3M steps) | Mac, automatic |
| At about 3M steps | Claude publishes checkpoint `ckpt-A` and restarts the Mac in phase B (adds real Freedoom maps) | Claude |
| Right after | Start Kaggle (section 2) | **You** |
| Overnight | Both computers train. Claude tests checkpoints every couple of hours on held-out maps. | automatic |
| Last 3 hours | Pick the best model, copy it to `submission/weights/`, validate, Codespaces docker check, brief PDF, push | Claude, plus you for the Codespaces check |
| Last 30 minutes | Final push. Submit the repo link and brief PDF on the form. | **You** |

## 5. If something looks wrong
| You see | It means | Do |
|---|---|---|
| `pgrep` prints nothing | Training stopped | Tell Claude; it restarts from `latest.pt` with no progress lost |
| `sps` below 200 for a long time | The Mac is overheating or busy | Close other apps and keep the Mac on a hard, cool surface |
| Kaggle says "GPU quota exceeded" | Your weekly 30 h of free GPU is used up | Skip Kaggle; the Mac alone is enough |
| `validate.py` shows **FAIL** | Something in the submission is broken | Tell Claude **before** the deadline |
