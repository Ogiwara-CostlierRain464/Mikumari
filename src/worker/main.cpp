#include <iostream>
#include "worker/worker.h"

int main(){
  auto *worker =  new mikumari::Worker();
  worker->join();

  std::cout << "Worker Exsiting" << std::endl;
}