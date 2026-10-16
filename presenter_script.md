# Presenter script — slides 2 & 3

**tiny alexa · ~8 minutes total** (slide 2 ≈ 3½ min, slide 3 ≈ 4½ min)

Delivery notes:
- Slide 2: click the architecture diagram to play the pulse animation while you talk through the pipeline.
- Slide 3: the three charts (curves + both confusion matrices) open in a centered popup when clicked — pop one up when you reach the numbers.
- The "19 intents · 18 slot values" phrase in the Dataset table has a hover popover — a good place to pause and let people read it.
- Numbers below are the real ones from `results/` — if someone asks, they check out.

---

## SLIDE 2 — The Architecture

So — the whole system in one picture. *(click the diagram)* It starts with you saying something — "set an alarm for 6 AM" — into the Pi's mic at 16 kilohertz. Everything after that is deterministic: the audio becomes a log-mel spectrogram, 64 frequency bands at 100 frames per second, so a 4.8-second command is a 64-by-480 grid. That grid goes through the model, and out the other side come two answers: an intent and a slot. *Alarm*, and *6 AM*. No cloud, no transcription, no text anywhere in the middle.

Now you might ask: why a TCN? Why not an LSTM, or a transformer, or just a regular CNN?

Here's the constraint that decided it: this has to run in single-digit milliseconds on a Raspberry Pi 4 — one CPU core, no GPU. That one rule kills most of the options.

An LSTM is sequential by definition — frame fifty has to wait for frame forty-nine. You can't parallelize over time, and on a Pi's CPU that's slow. And honestly, we don't need it: this isn't a streaming problem. The whole command is in hand before we act, so there's no benefit to processing it left to right.

A transformer would work, but the attention machinery — positional encodings, multi-head attention, softmax over a sequence — is a lot of moving parts for a fixed-length classifier with 480 frames. It'd be fine. It'd just be bigger and slower, for zero benefit here.

A plain CNN with small kernels only sees a narrow slice of time. To see the whole command you'd have to stack a lot of layers.

A TCN gets the wide view cheaply. Look at the five blocks — *(point)* — the dilation doubles each time: 1, 2, 4, 8, 16. Every conv still has a 5-frame kernel, but at dilation 16 one step reaches across 31 frames. Stack five of those and the receptive field is 1,005 frames — *more* than the 480-frame input. So by the time we pool, every unit has literally seen the entire command. And convolutions are embarrassingly parallel — no recurrence — which is exactly what a CPU wants.

Two more choices worth pointing out. First, the two stride-2 convs at the front: 480 goes to 240 goes to 120 frames. We downsample early so the dilated blocks work on a quarter of the time axis — that's where a lot of the Pi-side speed comes from. Second, the two heads at the end: one picks the intent out of 19, the other picks the slot out of 19 — that's 18 real values plus NO_SLOT, because most commands don't take a slot. And we don't just glue the two answers together independently — we decode jointly over the 31 intent-slot pairs that actually exist. The model can never say "weather at 6 AM," because that pair isn't in the table.

And why 4.8 seconds? I checked the actual data: after silence trimming, the longest command in the whole set is 3.0 seconds — the p99 is also 3.0, so that's not one outlier, that's the ceiling. 4.8 seconds gives headroom on top of that: slow speakers, the time-shift augmentation I add in training (up to 250 ms of lead-in), and the fact that the Pi's silence trim isn't sample-exact. And fixed length matters: every batch is a perfect rectangle — no padding, no masking — and the Pi's capture buffer is exactly 4.8 seconds, so training and live inference see identical shapes. Doubling the window would double the compute on the Pi, for a command that doesn't exist. So: 4.8 seconds.

---

## SLIDE 3 — Model & Results

Now the numbers. *(click a chart)* Left: label accuracy and validation loss over training. Right: the confusion matrices — intent and slot. Let me tell you how it got trained, and then I'll answer the question I know is coming: why did this only take 9 minutes?

First, the recipe. 14,256 training utterances, batch size 64, so 223 steps per epoch. The loss is just two cross-entropies — one per head — with a little label smoothing. AdamW, two-epoch warmup, cosine decay. Nothing exotic.

Three things I actually tuned, though. One: my dataset is 84 foreign speakers and 16 Filipino-English speakers, and I care about the second group as much as the first — so I oversample the Filipino speakers 2× during training, and I pick the best checkpoint on the *average* of the two groups' validation accuracy, not the overall number. Otherwise those 16 speakers get drowned out. Two: augmentation — speed perturbation, time shifts, and half the time I mix in white or pink noise at 5 to 30 dB SNR, plus SpecAugment. The reason that matters: the test split is half noisy recordings, and clean and noisy accuracy came out *identical* — 99.66% both ways. That's the augmentation doing its job. Three: early stopping, patience 10, on that group-balanced validation accuracy. Best epoch was 29; it stopped itself at 39.

And the result: 99.66% end-to-end on 1,789 test utterances from speakers the model never saw in training — the split is by speaker, 80/10/10, so this is generalization, not memorization. Slot head: 100%. Intent: 99.66% — and if you look at the confusion matrix, the handful of misses are the genuinely confusable ones: lights on versus off, "call," "time." Short commands with similar onsets.

Now — the 9 minutes. *(pause)*

I've been asked this, so I'll answer it honestly. 8,697 steps in 557 seconds is about 64 milliseconds per step. There are three reasons, and I'll rank them.

Number one: the model is small. 230,000 parameters, 64 channels, and after the early downsampling the sequence is only 120 frames long. Each step is a handful of small convolutions. There's simply not much compute in it.

Number two: the hardware. I trained on my MacBook — Apple Silicon, the MPS backend. No cloud GPU, no queue. For a model this size, that's plenty.

Number three — and this is the fun one — the DataLoader.

Here's the embarrassing part: my first version didn't have one. It was a for loop. For each batch: read the WAV, trim the silence, compute the log-mel, apply the augmentation, then forward and backward — all in one process. And the problem is that feature extraction is CPU work, and it's the slow part. Every step, the GPU sat there idle while the CPU did FFTs and mel filterbanks.

So I switched to a PyTorch DataLoader with four worker processes. Now the workers read the audio and compute the features for the *next* batches while the GPU trains on the *current* one — the two pipelines overlap. I also made the workers persistent, because on macOS spawning those processes is slow, and I wasn't about to pay that cost 39 times.

What does that actually buy you? In the for-loop version, wall-clock time is the *sum*: feature time plus training time, every single batch. With the DataLoader it's the *max* of the two — whichever is slower sets the pace, because they run in parallel. Do the back-of-envelope: the feature path — load, trim, perturb, noise, FFT, mel, SpecAugment — runs at about 1.3 milliseconds per utterance on this machine, so a batch of 64 costs roughly 80 milliseconds of serialized CPU work. That's the *same order* as the 64-millisecond training step. Serially, you pay both, every step: 8,697 steps × ~150 ms ≈ 21 minutes — more than double. Overlapped, the CPU work hides underneath the training and you pay the longer of the two, not the sum. And that's the optimistic case — on a loaded machine, or with slower disks, the CPU side grows and the gap widens. The DataLoader is what guarantees the GPU never waits for it.

And to be fair to the for loop: the DataLoader didn't conjure 9 minutes out of nothing. If the model were a 100-million-parameter transformer, great data loading wouldn't save you. It's the combination — a tiny model, fast silicon, and data that never makes the GPU wait. The DataLoader is just the piece that makes sure the other two are actually being used.

---

## Pocket Q&A

**"Why not just transcribe with Whisper and parse the text?"**
Whisper is hundreds of megabytes and takes seconds on the Pi — and it adds an error cascade: one misheard word breaks the parse. For a closed set of 31 commands, going straight from audio to intent skips the middleman and is both faster and more robust. If the vocabulary grew to thousands of phrases, I'd revisit.

**"Why 19 slot classes when there are 18 slot values?"**
The 19th class is NO_SLOT. Thirteen of the 19 intents take no slot, so the slot head has to be able to say "nothing" — and letting the softmax pick NO_SLOT is cleaner than special-casing it outside the head.

**"What happened to the 749 excluded files?"**
QC. I transcribed everything with an automatic transcriber and compared against the expected text; 749 takes were clipped, mispronounced, or mislabeled. They're kept in a FLAGGED folder so the exclusions are auditable, not silent.

**"Is the test set really unseen?"**
Split by speaker, not by file — 80/10/10 of the 100 speakers. The test speakers never appear in training.

**"Why symmetric padding in the TCN blocks instead of causal?"**
Causal padding exists for streaming — output at time t only uses inputs up to t. We pool the whole sequence before decoding, so order doesn't matter; causal padding would just waste compute.

**"Could you train on the Pi itself?"**
Technically yes — it's 0.9 MB. But the Pi is tens of times slower per step. I train once on the Mac, export ONNX, and ship the file. The Pi only ever runs inference.

**"What if I add more commands later?"**
The architecture scales fine to dozens or a hundred closed-set commands. Where it breaks is open vocabulary — "call Mom" versus "call Dad" — and at that point you want a real ASR front end, not more classes.

**"So the 9 minutes — is the DataLoader the whole story?"**
No — and I'll say it again: small model, fast Apple silicon, and overlapped data loading. The DataLoader's job is to make sure the GPU never waits on the CPU; the other two make each step cheap.
