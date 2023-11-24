#ifndef _MIKUMARI_GENERATOR_H_
#define _MIKUMARI_GENERATOR_H_
#include <map>
#include <random>
#include <unistd.h>
#include <libgen.h>
#include <dmlc/logging.h>
#include <sys/stat.h>
#include <__filesystem/directory_iterator.h>
#include <lz4.h>

#include "model_loader.h"

namespace mikumari {

inline bool exists(std::string filename) {
  struct stat buffer;
  return (stat (filename.c_str(), &buffer) == 0);
}

inline std::string get_base_directory()
{
  int bufsize = 1024;
  char buf[bufsize];
  int len = readlink("/proc/self/exe", buf, bufsize);
  return dirname(dirname(buf));
}

class InputGenerator {
private:
  std::minstd_rand rng;

  char *all_inputs;
  size_t all_inputs_size;

  std::map<size_t, std::vector<std::string>> compressed_inputs;
  std::map<size_t, std::vector<std::string>> uncompressed_inputs;

public:
  InputGenerator() {
    std::string base_dir = get_base_directory() +
      "/resources/inputs/processed";
    CHECK(exists(base_dir)) << "Could not find Clockwork images directory";

    size_t total_size = 0;
    std::vector<std::string> all_inputs;
    for (auto &p : std::filesystem::directory_iterator(base_dir)) {
      if (std::filesystem::is_directory(p)) {
        for (auto &f : std::filesystem::directory_iterator(p)) {
          std::string data;
          readFileAsString(f.path().u8string(), data);
          all_inputs.push_back(data);
          total_size += data.size();
          uncompressed_inputs[data.size()].push_back(data);


          const int max_compress_size = LZ4_compressBound(data.size());
          char* compressed = static_cast<char*>(malloc((size_t) max_compress_size));
          const int compressed_size = LZ4_compress_default(data.data(), compressed, data.size(), max_compress_size);
          std::string compressed_data = std::string(compressed, compressed_size);
          compressed_inputs[data.size()].push_back(compressed_data);
        }
      }
    }

    this->all_inputs = new char[total_size];
    size_t offset = 0;
    for (auto &input : all_inputs) {
      std::memcpy(this->all_inputs+offset, input.data(), input.size());
      offset += input.size();
    }
  }

  void generateInput(size_t size, char* buf) {
    auto it = uncompressed_inputs.find(size);
    if(it != uncompressed_inputs.end()) {
      auto &inputs = it->second;
      std::uniform_int_distribution<> d(0, inputs.size()-1);
      auto &input = inputs[d(rng)];
      CHECK(input.size() == size);
      std::memcpy(buf, input.data(), input.size());
    }else if(size < this->all_inputs_size) {
      std::uniform_int_distribution<> d(0, all_inputs_size - size - 1);
      size_t start_offset = d(rng);
      std::memcpy(buf, all_inputs + start_offset, size);
    }
  }
};

}

#endif