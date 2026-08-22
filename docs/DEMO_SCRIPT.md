# Demo Recording Script

A ~4-minute narrated walkthrough script, timed to match commands you actually run on screen. Read it in your own words rather than verbatim — the timings are a guide, not a script to memorize word-for-word.

## Recording setup (do this before hitting record)

1. **Tool:** Windows 11 has a built-in recorder (`Win + Alt + R` via Xbox Game Bar) — good enough for a single-window screen capture with mic audio. For more control (webcam corner, multiple windows, trimming), use **OBS Studio** (free) instead.
2. **Terminal font size:** bump to 16–18pt before recording — small terminal text is the #1 thing that makes recordings unwatchable on a phone screen.
3. **Two windows ready, not yet open:** a WSL2 terminal at the project root, and a browser tab pointed at `http://localhost:8000/` (don't load it yet — you want to show it connecting live).
4. **Pre-flight check**, run this once *before* recording so you're not debugging on camera:
   ```bash
   cd /mnt/c/Users/Aayush/fraud-detection
   sudo service docker start
   source scripts/wsl_env.sh
   python scripts/check_environment.py   # confirm everything shows AVAILABLE
   ```
5. **Close everything else** — Slack, email, notification popups. Do Not Disturb on.
6. **Resolution:** record at 1920x1080 minimum if your screen supports it; FAANG application portals and reviewers often watch on a laptop screen, and small text won't survive downscaling.

## The script

### 0:00–0:20 — Open on the problem, not the tech

> "I built an end-to-end fraud detection system to go deep on one specific engineering question: when does GPU acceleration actually pay off, and when does it not? Rather than assume the answer, I benchmarked five different inference backends — CPU, PyTorch GPU, a custom CUDA kernel I wrote, a C++ engine, and TensorRT — against each other, on the same hardware, the same model, the same methodology. Some of the results surprised me, and I'll show you why."

### 0:20–0:50 — Prove the environment is real

Run:
```bash
python scripts/check_environment.py
```

> "This isn't a claim — it's a live check. It's actually querying nvidia-smi, actually importing torch and checking CUDA availability, actually checking for TensorRT, the C++ compiler, CMake, Docker. Everything you see here is real state in this environment right now, not hardcoded text."

Let it finish printing — the RTX 3060, driver version, CUDA 11.3, TensorRT 10.0.1, all real.

### 0:50–1:50 — Terminal demo: real predictions, four backends

Run:
```bash
bash scripts/demo.sh
```

> "This loads the actual trained model — a small neural net, 4,033 parameters, trained on the real ULB credit card fraud dataset — and runs it through four different inference implementations on the same real, labeled transactions pulled from the held-out test set. Watch the backend column and the latency column."

While it prints:

> "Every one of these is a different code path I wrote and verified independently — the custom CUDA kernel fuses what's normally six separate GPU operations into one, and the C++ engine has zero Python or PyTorch dependency at runtime, it's calling the CUDA kernel directly and manages its own GPU memory."

Point at the final summary line (all correct).

> "All four backends agree with the ground truth on every transaction — that's not just a performance comparison, it's a correctness guarantee across five completely different implementations of the same math."

### 1:50–2:40 — Web demo, live click-through

Switch to browser, load `http://localhost:8000/`.

> "This page is served by the same FastAPI server the terminal was just talking to. Nothing here is client-side fake data."

Click **Load Fraud Example**.

> "This button just called GET /samples on the real API, which pulled a real transaction from the held-out test set — you can see the 29 feature values, which are the actual anonymized PCA components from the original dataset."

Click **Run Inference**.

> "And this POSTs those values to the real /predict endpoint. Fraud probability, the classification, which backend served it, and the latency — all real, all measured server-side in that request."

Click **Load Legitimate Example** → **Run Inference** to show the contrast.

### 2:40–3:20 — The honest benchmark story

Pull up the benchmark chart (`benchmarks/results/figures/latency_across_batch_sizes.png`) or the README section.

> "Here's the part I think is actually the most interesting engineering result in this project. I expected GPU to just win. It didn't — not at every batch size, and not uniformly. TensorRT, which is NVIDIA's own production inference optimizer, was actually the *slowest* GPU path of the five, at every single batch size I tested. I didn't just accept that — I built a second, fully static-shape TensorRT engine specifically to rule out that it was a configuration mistake, and got the same result. The real explanation is that TensorRT's runtime overhead isn't amortized by a model this small — it's built for production-scale models with millions of parameters, and mine has four thousand."

> "My own hand-written CUDA kernel, on the other hand, beat PyTorch's own GPU dispatch by over 2x at every batch size, because I profiled the actual bottleneck first — it turned out to be kernel launch overhead, not raw compute — and fused six kernel launches into one specifically to fix that."

### 3:20–3:50 — Engineering depth, briefly

> "A few things worth mentioning about how this was actually built: I hit a real compatibility wall trying to compile CUDA natively on Windows — the installed MSVC toolchain was incompatible with the CUDA 11.3 toolkit this driver needed — so I moved the GPU work into WSL2, which sidesteps that entirely. TensorRT itself was flagged as a risk going in, since it usually needs an NVIDIA developer account — I found a way around that too, using a specific CUDA-11-tagged package that resolves without any login. And building the verification tooling for this project actually caught two real bugs before they shipped — a PyTorch version incompatibility and a Python namespace collision with my own project directory."

### 3:50–4:00 — Close

> "Everything I just showed is in the GitHub repo, with the full commit history, a test suite of 68 automated tests, and documentation for every phase including the dead ends. Thanks for watching."

## After recording

- **Trim dead air** at the start/end — most editors (even Windows Photos' basic trim, or CapCut free) handle this in under a minute.
- **Export at 1080p, mp4** — universally playable.
- **Don't over-produce.** A clean, well-narrated screen recording reads as more credible than a heavily edited one for this purpose — the goal is proof the system works, not a marketing reel.
- **Where to host:** an unlisted YouTube link is the most universally compatible option for application portals; Loom works too if the portal accepts embedded links rather than file uploads.
