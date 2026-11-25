// Minimal CUPTI injection library (spike, Task 7). Loaded via CUDA_INJECTION64_PATH.
// Enables CONCURRENT_KERNEL and COMPUTE_ENGINE_CTX_SWITCH, counts records, prints totals to stderr at exit.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>

static const size_t kBufSize = 1 << 20;
static std::mutex g_mu;
static uint64_t g_kernel = 0, g_ctxsw = 0, g_ctxsw_start = 0, g_ctxsw_end = 0, g_other = 0;
static uint64_t g_dropped = 0;
static std::set<uint32_t> g_ctxsw_ctx, g_kernel_ctx;

static void CUPTIAPI bufferRequested(uint8_t **buf, size_t *size, size_t *maxRecords) {
  *size = kBufSize;
  *maxRecords = 0;
  *buf = (uint8_t *)aligned_alloc(8, kBufSize);
}

static void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t *buf, size_t, size_t validSize) {
  CUpti_Activity *rec = nullptr;
  std::lock_guard<std::mutex> lk(g_mu);
  while (validSize > 0 &&
         cuptiActivityGetNextRecord(buf, validSize, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      g_kernel++;
      g_kernel_ctx.insert(((CUpti_ActivityKernel11 *)rec)->contextId);
    } else if (rec->kind == CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH) {
      auto *r = (CUpti_ActivityComputeEngineCtxSwitch *)rec;
      g_ctxsw++;
      if (r->operationType == CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_START) g_ctxsw_start++;
      if (r->operationType == CUPTI_COMPUTE_ENGINE_CTX_SWITCH_OPERATION_END) g_ctxsw_end++;
      g_ctxsw_ctx.insert(r->contextId);
    } else {
      g_other++;
    }
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped) == CUPTI_SUCCESS) g_dropped += dropped;
  free(buf);
}

static void report() {
  cuptiActivityFlushAll(1);
  std::lock_guard<std::mutex> lk(g_mu);
  fprintf(stderr,
          "[cupti-inject] kernel_records=%llu ctxsw_records=%llu (start=%llu end=%llu) other=%llu dropped=%llu "
          "distinct_ctx_in_ctxsw=%zu distinct_ctx_in_kernels=%zu\n",
          (unsigned long long)g_kernel, (unsigned long long)g_ctxsw, (unsigned long long)g_ctxsw_start,
          (unsigned long long)g_ctxsw_end, (unsigned long long)g_other, (unsigned long long)g_dropped,
          g_ctxsw_ctx.size(), g_kernel_ctx.size());
  // CUPTI exposes only contextId (no process id) in the ctx-switch record.
  for (uint32_t c : g_ctxsw_ctx) fprintf(stderr, "[cupti-inject] ctxsw contextId=%u\n", c);
  for (uint32_t c : g_kernel_ctx) fprintf(stderr, "[cupti-inject] kernel contextId=%u\n", c);
}

static void tryEnable(CUpti_ActivityKind kind, const char *name) {
  CUptiResult r = cuptiActivityEnable(kind);
  const char *msg = nullptr;
  if (r != CUPTI_SUCCESS) cuptiGetResultString(r, &msg);
  fprintf(stderr, "[cupti-inject] enable %s: %s\n", name, r == CUPTI_SUCCESS ? "OK" : (msg ? msg : "error"));
}

extern "C" int InitializeInjection(void) {
  uint32_t ver = 0;
  cuptiGetVersion(&ver);
  fprintf(stderr, "[cupti-inject] loaded, CUPTI runtime version %u (header %d)\n", ver, CUPTI_API_VERSION);
  cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  tryEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "CONCURRENT_KERNEL");
  tryEnable(CUPTI_ACTIVITY_KIND_COMPUTE_ENGINE_CTX_SWITCH, "COMPUTE_ENGINE_CTX_SWITCH");
  atexit(report);
  return 1;
}
