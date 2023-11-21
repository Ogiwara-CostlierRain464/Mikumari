#include <iostream>

int main(){
  auto *worker =  new Worker();
  worker->join();

  std::cout << "Worker Exsiting" << std::endl;
}