#ifndef _CLOCKWORK_API_WORKER_API_H_
#define _CLOCKWORK_API_WORKER_API_H_

#include <vector>
#include <functional>
#include <string>
#include <memory>

/**
This is the API for Clockwork Workers that are controlled by a centralized Clockwork scheduler.

This API is ONLY used by the clockwork runtime, which expects to receive specific actions to
execute at specific times.
*/

const int actionSuccess = 0; // Action completed successfully
const int actionCancelled = 1; // Action cancelled for some other reason
const int actionErrorRuntimeError = 2; // Action cancelled due to runtime error
const int actionErrorShuttingDown = 3; // Action cancelled due to clockwork shutting down

const int actionErrorUnknownModel = 10; // Action requested an unknown model
const int actionErrorCouldNotStartInTime = 11; // Action dropped because it could not be executed in time
const int actionErrorInvalidAction = 12; // An invalid action type was specified
const int actionErrorInvalidGPU = 13; // An invalid GPU device id was specified

const int actionErrorModelWeightsNotPresent = 20; // Infer or Evict action could not happen because no weights
const int actionErrorWeightsAlreadyLoaded = 21; // LoadWeightsAction failed because weights already loaded
const int actionErrorWeightsInUse = 22; // LoadWeightsAction failed because weights are being actively used (e.g. for transfer)
const int actionErrorWeightsChanged = 23; // Infer action failed because weights changed while executing

const int actionErrorInvalidInput = 30; // Invalid input to an inference action
const int actionErrorInvalidBatchSize = 31; // Invalid batch size input to an inference action

const int actionErrorInvalidModelID = 40; // Invalid ID specified for load model
const int actionErrorInvalidModelPath = 41; // Invalid path specified for load model

const int networkSendTooLate = 45; // Couldn't start netsend from controller->worker in time

const int loadWeightsTooEarly = 50;
const int loadWeightsTooLate = 51;
const int loadWeightsUnknownModel = 52;
const int loadWeightsInsufficientCache = 53;
const int loadWeightsConcurrentModification = 54;

const int evictWeightsTooEarly = 60;
const int evictWeightsTooLate = 61;
const int evictWeightsUnknownModel = 62;
const int evictWeightsNotInCache = 63;

const int copyInputTooEarly = 70;
const int copyInputTooLate = 71;
const int copyInputUnknownModel = 72;
const int copyInputInvalidBatchSize = 73;
const int copyInputHostAlloc = 74;
const int copyInputInvalidInput = 75;
const int copyInputIOPoolExhausted = 76;
const int copyInputBadSizes = 77;
const int copyInputBadDecompress = 78;

const int execTooEarly = 80;
const int execTooLate = 81;
const int execWeightsMissing = 82;
const int execWorkspacePoolExhausted = 83;
const int execConcurrentWeightsModification = 84;

const int copyOutputTooEarly = 90;
const int copyOutputTooLate = 91;
const int copyOutputHostAlloc = 92;







namespace clockwork {
namespace workerapi {

/* Action types */
const int loadModelFromDiskAction = 0;
const int loadWeightsAction = 1;
const int inferAction = 2;
const int evictWeightsAction = 3;
const int clearCacheAction = 4;
const int getWorkerStateAction = 5;

class Action {
public:
	int id;
	int action_type;

	// The below are not sent over the network
	int64_t clock_delta = 0; // Estimated clock delta between controller and this worker
	uint64_t received = 0;
	uint64_t action_sent = 0;

	virtual std::string str() = 0;
};

class LoadModelFromDisk : public Action {
public:
	int model_id;
	int no_of_copies;
	std::string model_path;
	uint64_t earliest;
	uint64_t latest;
	unsigned max_batch_size = 32;
	uint64_t max_exec_duration = 1000000000UL;

	virtual std::string str();
};

class LoadWeights : public Action {
public:
	uint64_t earliest;
	uint64_t latest;
	uint64_t expected_duration = 0;

	int model_id;
	unsigned gpu_id;

	// Not actually sent to workers; here for convenience
	int worker_id = -1;
	uint64_t expected_exec_complete = 0;

	virtual std::string str();
};

class EvictWeights : public Action {
public:
	uint64_t earliest;
	uint64_t latest;

	int model_id;
	unsigned gpu_id;

	// Not actually sent to workers; here for convenience
	int worker_id = -1;

	virtual std::string str();
};

class Infer : public Action {
public:
	uint64_t earliest;
	uint64_t latest;
	uint64_t expected_duration = 0;

	int model_id;
	unsigned gpu_id;
	int batch_size;
	int input_size;
	char* input;
	std::vector<size_t> input_sizes;

	// Not actually sent to workers; here for convenience
	int worker_id = -1;
	int expected_gpu_clock = 0;
	uint64_t expected_exec_complete = 0;

	virtual std::string str();
};

class ClearCache : public Action {
public:
	virtual std::string str();
};

class GetWorkerState : public Action {
public:
	virtual std::string str();
};

class Result {
public:
	int id;
	int action_type;
	int status;

	uint64_t action_received;
	uint64_t result_sent;

	// Not sent over the network
	uint64_t result_received = 0;
	int64_t clock_delta = 0; // Estimated clock delta between controller and this worker

	virtual std::string str() = 0;
};

class ErrorResult : public Result {
public:
	std::string message;

	virtual std::string str();
};

class Timing {
public:
	uint64_t begin;
	uint64_t end;
	uint64_t duration; // For async tasks this is NOT end-begin

	virtual std::string str();
};

class ModelInfo {
public:
	int id;
	std::string source; // Useful for experiments but not strictly necessary
	size_t input_size;
	size_t output_size;
	std::vector<unsigned> supported_batch_sizes;
	size_t weights_size;
	unsigned num_weights_pages;
	uint64_t weights_load_time_nanos;
	std::vector<uint64_t> batch_size_exec_times_nanos;

	virtual std::string str();
};

class LoadModelFromDiskResult : public Result, public Timing {
public:
	// TODO: use ModelInfo instead of putting it all here.
	size_t input_size;
	size_t output_size;
	int copies_created;
	std::vector<unsigned> supported_batch_sizes;

	size_t weights_size_in_cache;
	unsigned num_weights_pages;

	// If measurements exist, they will be populated here; otherwise 0
	uint64_t weights_load_time_nanos;
	std::vector<uint64_t> batch_size_exec_times_nanos;

	virtual std::string str();
};

class LoadWeightsResult : public Result, public Timing {
public:

	virtual std::string str();
};

class EvictWeightsResult : public Result, public Timing {
public:

	virtual std::string str();
};

class InferResult : public Result {
public:
	Timing copy_input;
	Timing exec;
	Timing copy_output;
	int output_size;
	char* output;
	unsigned gpu_id;
	unsigned gpu_clock_before;
	unsigned gpu_clock;

	virtual std::string str();
};

class ClearCacheResult : public Result {
public:
	virtual std::string str();
};

class GPUInfo {
public:
	int id;
	size_t weights_cache_size;
	unsigned weights_cache_total_pages;
	std::vector<unsigned> models; // Models currently on GPU
	size_t io_pool_size; // Not actually useful but included for completeness
	size_t workspace_pool_size; // Not actually useful but included for completeness

	virtual std::string str();
};

class WorkerMemoryInfo {
public:
	size_t page_size;
	size_t host_weights_cache_size; // Host doesn't actually have a fixed-size cache at the moment
	size_t host_io_pool_size; // Not actually useful but included for completeness
	std::vector<GPUInfo> gpus;
	std::vector<ModelInfo> models;

	virtual std::string str();
};

class GetWorkerStateResult : public Result {
public:
	WorkerMemoryInfo worker;

	virtual std::string str();
};

// TODO: upload model action or RPC possibly

}
}

#endif
