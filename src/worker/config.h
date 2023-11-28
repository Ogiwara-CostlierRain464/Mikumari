#ifndef CONFIG_H
#define CONFIG_H

namespace mikumari {

class WorkerConfig {
public:
  unsigned num_gpus = 1;

  size_t weights_cache_size = 13'421'772'800L;
  size_t weights_cache_page_size = 16'777'216L;
  size_t io_pool_size = 536'870'912L;
  size_t workspace_pool_size = 536'870'912L;
  size_t host_io_pool_size = 536'870'912L;

  bool allow_zero_size_inputs = true;

  WorkerConfig()= default;
};

}

#endif //CONFIG_H
