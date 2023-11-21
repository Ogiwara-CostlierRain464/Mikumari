#ifndef WORKER_H
#define WORKER_H

#include <chrono>

namespace Mikumari {


class DummyInfer {
public:
  std::chrono::steady_clock::time_point start;

  void run() {
    start = std::chrono::steady_clock::now();

    // Load model

    // Copy Input

    // Do Infer
  }
};

class Worker {
public:
  void exec() {
    // do infer
  }
};

}

#endif //WORKER_H
