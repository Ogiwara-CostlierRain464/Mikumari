#ifndef _CLOCKWORK_UTIL_H_
#define _CLOCKWORK_UTIL_H_

#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <deque>
#include <random>

#define NUM_GPUS_1 1
#define NUM_GPUS_2 2
#define GPU_ID_0 0

namespace clockwork {

typedef std::chrono::steady_clock::time_point time_point;

namespace util {

// High-resolution timer, current time in nanoseconds
std::uint64_t now();
std::string millis(uint64_t t);

time_point hrt();

std::uint64_t nanos(time_point t);

std::string nowString();

unsigned get_num_gpus();

void setCudaFlags();

std::string getGPUmodel(int deviceNumber);

extern "C" char* getGPUModelToBuffer(int deviceNumber, char* buf);

void printCudaVersion();


void readFileAsString(const std::string &filename, std::string &dst);
std::vector<std::string> listdir(std::string directory);
bool exists(std::string filename);
long filesize(std::string filename);


void initializeCudaStream(unsigned gpu_id = 0, int priority = 0);
void SetStream(cudaStream_t stream);
cudaStream_t Stream();

// A hash function used to hash a pair of any kind
// Source: https://www.geeksforgeeks.org/how-to-create-an-unordered_map-of-pairs-in-c/
struct hash_pair {
	template <class T1, class T2>
	size_t operator()(const std::pair<T1, T2>& p) const {
		auto hash1 = std::hash<T1>{}(p.first);
		auto hash2 = std::hash<T2>{}(p.second);
		return hash1 ^ hash2;
	}
};

struct hash_tuple {
	template <class T1, class T2, class T3>
	size_t operator()(const std::tuple<T1, T2, T3>& t) const {
		auto hash1 = std::hash<T1>{}(std::get<0>(t) );
		auto hash2 = std::hash<T2>{}(std::get<1>(t) );
		auto hash3 = std::hash<T3>{}(std::get<2>(t) );
		return hash1 ^ hash2 ^ hash3;
	}
};

std::string get_clockwork_directory();

std::string get_example_model_path(std::string model_name = "resnet18_tesla-m40");

std::string get_example_model_path(std::string clockwork_directory, std::string model_name);

std::string get_controller_log_dir();
std::string get_modelzoo_dir();
std::string get_clockwork_model(std::string shortname);

std::map<std::string, std::string> get_clockwork_modelzoo();
bool client_inputs_disabled();

class InputGenerator {
 private:
	std::minstd_rand rng;

 	char* all_inputs;
 	size_t all_inputs_size;

 	std::map<size_t, std::vector<std::string>> compressed_inputs;
 	std::map<size_t, std::vector<std::string>> uncompressed_inputs;


 public:
 	InputGenerator();

 	void generateInput(size_t size, char* buf);
 	void generateInput(size_t size, char** bufPtr);
 	void generateCompressedInput(size_t size, char** bufPtr, size_t* compressed_size);
 	void generatePrecompressedInput(size_t size, char** bufPtr, size_t* compressed_size);
 	std::string& getPrecompressedInput(size_t size);
};


#define DEBUG_PRINT(msg) \
	std::cout << __FILE__ << "::" << __LINE__ << "::" << __FUNCTION__ << " "; \
	std::cout << msg << std::endl;

}
}


#endif
