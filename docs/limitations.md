# Limitations and how to read the numbers

gso shows what the GPU stack exposes and labels what it had to guess. This page lists
what each number does and does not mean. Read it before drawing conclusions from a
screen or a report.

Evidence for most statements here is in
[`verification/RESULTS.md`](verification/RESULTS.md) (written in Korean) and
[`verification/nsys-schema.md`](verification/nsys-schema.md). All measurements came from one
shared multi-GPU machine (RTX A5000, driver 580.126.20, nsys 2026.6.1), mostly
without root. Nothing was re-verified on other GPUs, drivers or nsys versions.

## General rules

- **Unsupported or unknown values are `N/A`, never `0`.** A field that the GPU, driver or
  host does not provide, or that could not be read, is shown as `N/A`. A real zero is shown as
  zero. For example, DCGM field 1004 (Tensor Active) reads 0.000 under a small-kernel load
  and about 0.95 under an FP16 matrix multiply on the test GPU; those are measurements, not
  gaps.
- **Measured, estimated and unsupported values are kept apart.** Estimates are labelled as
  estimates in the screens and reports.
- **`--gpu` is required.** There is no default GPU, on purpose: on a shared machine the
  wrong default would touch someone else's GPU.
- **Nothing is changed on your system.** gso never uses `sudo`, never modifies an MPS
  server or daemon that it did not start itself, and `gso bench` aborts if another process
  appears on the GPU.

## Device counters (DCGM)

- DCGM fields 1001 to 1005 are GR Engine Active, SM Active, SM Occupancy, Tensor Active and
  DRAM Active. They are **device-level ratios averaged over the sampling interval**, shown
  as percent.
- **Tensor Active** reports activity of the tensor pipes. It is not achieved FLOPS and not
  a count of physical Tensor Cores in use.
- **SM Active** is not the instruction issue rate. A GPU can be 99% "GR Engine Active"
  while SM Active is below 10% (small kernels did exactly that on the test machine).
- **DRAM Active** alone does not prove memory-bandwidth saturation.
- Support differs by GPU. gso asks `dcgmi profile -l` first and only requests supported
  fields, because one unsupported field makes the whole `dcgmi dmon` call fail. Field 1006
  was not supported on the RTX A5000.
- DCGM needs a running host engine. gso does not start one. If it is unreachable, all DCGM
  rows are `N/A` and `gso doctor` says why.

## Per-process numbers

- **Device counters cannot be split by PID.** DCGM counters belong to the device, not to a
  process. Any per-PID figure derived from time correlation is an **estimate**. gso does
  not claim an exact per-process share of Tensor or DRAM activity.
- The per-process SM utilisation and VRAM columns come from NVML, which has its own limits
  (no fine hardware counters per process).
- Without root, other users' processes appear masked in nsys output (see below).
  Containers use their own PID namespace, so PIDs seen inside and outside can differ.

## Context switching (nsys capture)

- **Capture is on-demand, not real time.** `gso capture` and the capture key in `gso live`
  run one Nsight Systems capture of 5 to 10 seconds and report on that window only. It is
  not a continuous or streaming view, and a capture can lose events (see "dropped events").
  A capture takes about 10 seconds plus export time and slows the workloads while it runs.
- **Two capture modes with different contents.**
  - Global capture (no command after `--`): context-switch events of every process on every
    GPU. No CUDA kernel intervals.
  - Launched capture (a command after `--`): the launched process's kernel intervals. Few
    context-switch events. You cannot get "two contending processes plus both kernel
    timelines" from one file with the tested nsys.
- **The context-interval constructions are estimates and they differ.** Switch events
  only say "RESTORE_START" and "SAVE_END" for a context. Two readings are possible and the
  data does not choose between them:
  - `restore_save`: the interval from a context's RESTORE_START to its next SAVE_END. On
    the test pair this was about 20% of the window per context.
  - `hold`: a context is assumed to hold the GPU from its RESTORE_START until another
    context's RESTORE_START. About 50% per context on the same data.

  gso shows both, labels both as estimates, and does not merge them into a single
  "residency" number. Do not read either as measured GPU occupancy.
- **Context-inactive time is not switching overhead.** The time a context does not hold
  the GPU includes waiting for work, other contexts' slices and idle time. The gap between
  intervals is not a switch cost either.
- **Switch rate is derived from RESTORE_START events** in the capture window and is a
  lower bound when events are missing. It is not a count of "real" hardware switches
  confirmed by another source.
- **Without root, other users' processes are masked.** Their events appear under one
  masked context with no PID. Timestamp precision for them may also be lower. Root capture
  was not run, so what root would show is **provisional**: the pass verdicts for the
  global-capture and MPS-capture questions in RESULTS (G2, G4) are marked provisional for
  this reason. `gso` itself never calls `sudo`.
- **PID decoding is an empirical rule.** The PID is read from the upper bits of nsys's
  `globalPid`. It matched the workload PIDs and every row of the `PROCESSES` table in the
  tested export, but it is not taken from nsys documentation.
- **Kernel queue delay is not available.** The kernel table has start and end times only.
  gso shows API-call-to-kernel-start delay as an estimate, which is not submission-to-start
  and mixes in earlier kernels when work is asynchronous.
- **Dropped events.** nsys writes no dedicated loss counter. gso reads the diagnostic
  messages and sequence-number gaps it can find. "No gap found" is not proof that nothing
  was dropped.
- **Capture window.** Intervals that are still open at the end of the window are cut at the
  window end, not discarded. That choice changes the ratios for short captures.
- **DCGM during a capture.** In one test run (n=1) SM Active, SM Occupancy and DRAM Active
  read about 15 to 18% lower while a capture was running. The cause (collector
  interference or the slower workload) was not separated. The live view keeps showing DCGM
  values during a capture and marks them "capturing" instead of replacing them with `N/A`.

## Kernel overlap

- **Kernel-interval overlap is not SM-level concurrency.** It is the time during which
  kernel start-to-end intervals of different PIDs intersect. It does not say how many SMs
  ran each kernel at the same time, nor that the kernels shared an SM.
- Overlap needs kernel intervals from both processes. A launched capture contains kernels
  of the launched process only, so for two independent processes it cannot be computed from
  one capture. The MPS A/B benchmark therefore reports application throughput and latency,
  not overlap.

## CUDA MPS

- **MPS detection.** A GPU counts as MPS ON when the NVML/`nvidia-smi` process list shows
  `nvidia-cuda-mps-server` or a process of type `M` / `M+C`. MPS control commands work only
  for the user who owns the control pipe, so they are not used for detection.
- **CTX SW/s is `N/A (MPS)` on an MPS GPU.** In the test run, MPS ON produced 0 context
  switch events on the GPU against about 205,000 with MPS OFF. Zero events cannot be told
  apart from "not recorded", so gso does not print `0/s`.
- **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is an upper bound.** It limits the execution
  resources a client may use. It is not an exclusive physical SM partition and not a
  guaranteed throughput. gso reports the value you **requested** for each client and the
  value the **server** reported, in separate columns. In the test runs the server value
  stayed 100 whatever was requested per client. gso does not claim a limit was "applied".
- **Set before clients start.** The percentage is read when a client starts. `gso bench`
  starts its own user-scoped MPS daemon for each MPS condition and stops it afterwards.
  It refuses to run when any MPS daemon or server already exists on the machine, and it
  does not change a production MPS server or the compute mode.
- **MPS ON can also make latency worse.** Sharing resources can improve throughput or
  hurt tail latency depending on the workload. The README example shows a large gain for
  small kernels; a TensorRT plus small-kernel pair in a separate verification run lost
  about 31% throughput on the TensorRT side.

## Benchmark (`gso bench` / `gso compare`)

- **Workloads are closed-loop** (they run as fast as they can). "Same request rate"
  comparisons are not supported yet.
- **Condition order is fixed**, not randomised, so GPU temperature drift is confounded
  with the condition. Start temperatures are recorded and shown in the report notes.
- **GPU clocks are not locked** (this needs privileges). gso waits for the GPU to cool
  between repetitions (at most 60 s); when that times out, the report says so.
- **Repetitions are few.** The minimum is 3. Coefficients of variation are shown, but
  small differences at n=3 should not be quoted as effects.
- **Contamination detection.** If another process appears on the GPU during a repetition,
  that repetition is marked unfair and excluded from the statistics, and the run aborts
  for safety. Excluded rows are counted in the report. Contamination can only be detected
  when the other process is visible to NVML.
- **Profiled and unprofiled rows must not be compared.** With `--profile`, each repetition
  also runs an on-demand capture, which slows the workloads.
- **Single GPU, two workloads.** No multi-GPU runs, no more than two clients.

## Overhead

Measured once, on one workload (two small-kernel processes, one GPU), with n=3 repetitions
per condition:

| Collector | Change in total throughput |
|---|---|
| DCGM sampling at 100 ms | +0.12% (inside run-to-run variation; do not quote as an effect) |
| nsys global capture, 10 s inside a 30 s run | -6.94% averaged over the 30 s (the effect inside the 10 s window is larger; it was not separated) |

Other workloads, other sampling rates and root captures were not measured.

## Platform

- Verified only on Linux x86_64 with RTX A5000 GPUs. Other GPUs, MIG, vGPU, and Jetson
  are untested; the CUPTI context-switch event is documented as unsupported on MIG, vGPU
  and QNX.
- Nsight Compute's PM-sampling context-switch trace is documented as unsupported with
  multiple MPS clients. gso does not use it, and this constraint is not generalised to
  other tracing APIs.
- Feature detection is per GPU and per driver. If something cannot be read, it is shown
  as `N/A` with a reason.
