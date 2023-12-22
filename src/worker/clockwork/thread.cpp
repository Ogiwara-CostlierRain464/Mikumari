#include "thread.h"

#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <algorithm>
#include <sstream>
#include <sched.h>

#include <dmlc/logging.h>

#include "cuda_common.h"
#include "util.h"

namespace clockwork {
namespace threading {

// The priority scheduler in use.  SCHED_FIFO or SCHED_RR
int scheduler = SCHED_FIFO;

int maxPriority(int scheduler) {
  return 49; // 1 less than interrupt priority
  // return sched_get_priority_max(scheduler);
}


void setPriority(int scheduler, int priority, pthread_t thId) {
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(scheduler);
  int ret = pthread_setschedparam(thId, scheduler, &params);
  CHECK(ret == 0) << "Unable to set thread priority.  Don't forget to set `rtprio` to unlimited in `limits.conf`.  See Clockwork README for instructions";

  int policy = 0;
  ret = pthread_getschedparam(thId, &policy, &params);
  CHECK(ret == 0) << "Unable to verify thread scheduler params";
  CHECK(policy == scheduler) << "Unable to verify thread scheduler params";
}

void setMaxPriority() {
  setPriority(SCHED_FIFO, maxPriority(SCHED_FIFO), pthread_self());
}


}
}