#include <iostream>
#include <boost/asio.hpp>
#include <boost/version.hpp>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <chrono>
#include <sstream>
#include <queue>

#include "clockwork/model/batched.h"
#include "clockwork/memory.h"
#include "clockwork/cache.h"
#include "clockwork/cuda_common.h"
#include "clockwork/thread.h"
#include "clockwork/util.h"

namespace asio = boost::asio;
using asio::ip::tcp;
using namespace std::chrono_literals;
using namespace clockwork;

model::BatchedModel* load_model(std::string model) {
    return model::BatchedModel::loadFromDisk(model, 0);
}

void check_model(int page_size, int iterations, std::string model_path, std::string inputfile, bool print_full_output, bool print_topn_output) {
    if (inputfile == "") {
        std::cout << "Checking " << model_path << " using random input" << std::endl;
    } else {
        std::cout << "Checking " << model_path << " using provided input " << inputfile << std::endl;
    }

    util::setCudaFlags();
    util::initializeCudaStream();
    threading::setMaxPriority();

    clockwork::model::BatchedModel* model = load_model(model_path);

    auto batch_sizes = model->implemented_batch_sizes();

    model->instantiate_models_on_host();
    for (auto &p : model->models) {
        p.second->rate_limit = false;
    }

    size_t weights_page_size = page_size;
    size_t weights_cache_size = model->num_weights_pages(weights_page_size) * weights_page_size;
    PageCache* weights_cache = make_GPU_cache(weights_cache_size, weights_page_size, GPU_ID_0);

    cudaError_t status;
    model->instantiate_models_on_device();

    unsigned num_pages = model->num_weights_pages(weights_page_size);
    std::shared_ptr<Allocation> weights = weights_cache->alloc(num_pages, []{});
    CHECK_NOTNULL(weights);

    int warmups = 10;

    // Time the transfer
    for (unsigned i = 0; i < warmups; i++) {
        model->transfer_weights_to_device(weights->page_pointers, util::Stream());
    }
    status = cudaStreamSynchronize(util::Stream());
    CHECK(status == cudaSuccess);
    auto before_transfer = util::now();
    for (unsigned i = 0; i < iterations; i++) {
        model->transfer_weights_to_device(weights->page_pointers, util::Stream());
    }
    status = cudaStreamSynchronize(util::Stream());
    CHECK(status == cudaSuccess);
    auto after_transfer = util::now();
    std::cout << "  input_size:  " << model->input_size(1) << std::endl;
    std::cout << "  output_size: " << model->output_size(1) << std::endl;
    std::cout << "  workspace:   " << model->workspace_memory_size(1) << std::endl;
    std::cout << "  weights size paged (non-paged) [num_pages]: " << (weights_page_size * num_pages) << " (" << model->weights_size << ") [" << num_pages << "]" << std::endl;
    printf("  weights transfer latency: %.2f ms\n", ((float) (after_transfer-before_transfer)) / (iterations * 1000000.0));
    std::cout << "  execution latency:" << std::endl;

    // Read in the input data if provided
    std::string inputdata;
    if (inputfile != "") {
        util::readFileAsString(inputfile, inputdata);

        if (inputdata.size() != model->input_size(1)) {
            std::cout << "Error with provided input " << inputfile
                      << ", expected size " << model->input_size(1)
                      << " but got " << inputdata.size() << std::endl;
            return;
        }
    }

    // For printing the output
    std::stringstream outputdata_string;
    std::stringstream outputdata_topn;

    for (unsigned batch_size : batch_sizes) {
        // Create inputs and outputs
        char* input = new char[model->input_size(batch_size)];
        char* output = new char[model->output_size(batch_size)];

        // Use the input data if it was provided
        if (inputfile != "") {
            for (unsigned i = 0; i < batch_size; i++) {
                std::memcpy(input + (i * inputdata.size()), inputdata.data(), inputdata.size());
            }
        }

        // Create and allocate io_memory on GPU
        size_t io_memory_size = model->io_memory_size(batch_size);
        MemoryPool* io_pool = CUDAMemoryPool::create(io_memory_size, GPU_ID_0);
        char* io_memory = io_pool->alloc(io_memory_size);

        // Create and allocate intermediate GPU memory workspace
        size_t workspace_size = model->workspace_memory_size(batch_size);
        MemoryPool* workspace_pool = CUDAMemoryPool::create(workspace_size, GPU_ID_0);
        char* workspace_memory = workspace_pool->alloc(workspace_size);

        // Now execute each step of model
        model->transfer_input_to_device(batch_size, input, io_memory, util::Stream());

        // Time the call
        for (int i = 0; i < warmups; i++) {
            model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, util::Stream());
        }
            status = cudaStreamSynchronize(util::Stream());
            CHECK(status == cudaSuccess);
        auto before = util::now();
        for (int i = 0; i < iterations; i++) {
            model->call(batch_size, weights->page_pointers, io_memory, workspace_memory, util::Stream());
        }
            status = cudaStreamSynchronize(util::Stream());
            CHECK(status == cudaSuccess);
        auto after = util::now();
        printf("    b%d: %.2f ms\n", batch_size, ((float) (after-before)) / (iterations * 1000000.0));

        model->transfer_output_from_device(batch_size, output, io_memory, util::Stream());

        status = cudaStreamSynchronize(util::Stream());
        CHECK(status == cudaSuccess);

        // We will print the batchsize 1 output
        if (batch_size == 1) {
            // Full output string
            float* outputf = static_cast<float*>(static_cast<void*>(output));
            unsigned output_size = model->output_size(batch_size)/4;
            for (unsigned i = 0; i < output_size; i++) {
                outputdata_string << outputf[i] << " ";
            }

            // Summary output string
            std::priority_queue<std::pair<float, unsigned>> q;
            for (unsigned i = 0; i < output_size; i++) {
                q.push(std::pair<float, unsigned>(outputf[i], i));
            }
            int topn = 5;
            for (unsigned i = 0; i < topn; i++) {
                outputdata_topn << "output[" << q.top().second << "] = " << q.top().first << std::endl;
                q.pop();
            }
        }


        delete input;
        delete output;

        io_pool->free(io_memory);
        delete io_pool;

        workspace_pool->free(workspace_memory);
        delete workspace_pool;
    }

    weights_cache->unlock(weights);
    weights_cache->free(weights);
    delete weights_cache;

    model->uninstantiate_models_on_device();
    model->uninstantiate_models_on_host();

    delete model;

    if (print_full_output) {
        std::cout << "Output data:" << std::endl;
        std::cout << outputdata_string.str() << std::endl;
    }
    if (print_topn_output) {
        std::cout << "Output data topn:" << std::endl;
        std::cout << outputdata_topn.str() << std::endl;
    }
}


class Session;

tbb::concurrent_bounded_queue<std::shared_ptr<Session>> queue{};

class Session : public std::enable_shared_from_this<Session>{
public:
  tcp::socket socket;

  explicit Session(tcp::socket &&socket):
  socket(std::move(socket)){}

  void start() {
    asio::streambuf buf;
    boost::system::error_code err;
    asio::read_until(socket, buf, "\0", err);

    if(err && err != asio::error::eof) {
      std::cerr << "receive failed: " << err.message() << std::endl;
    }else {
      auto data = asio::buffer_cast<const char*>(buf.data());
      std::cout << "ok, I use " << data << " model" << std::endl;

      queue.push(shared_from_this());
    }
  }
};

// call GPU at here
void read_thread() {
  for(;;) {
    std::shared_ptr<Session> req;
    queue.pop(req);

    // call gpu at here
    check_model(16 * 1024 * 1024, 100, "../resources/resnet18_tesla-m40/model", "", false, false);

    std::string msg = "cat detected";
    boost::system::error_code err;
    asio::write(req->socket, asio::buffer(msg), err);
    req->socket.close();
  }
}


class Server {
  tcp::acceptor acceptor;

public:
  explicit Server(asio::io_context &io_context, unsigned short port):
    acceptor(io_context, tcp::endpoint(tcp::v4(), port))
  {}

  void doAccept() {
    acceptor.async_accept([this](const boost::system::error_code &err, tcp::socket socket) {
      if(err) {
        std::cerr << err.message() << std::endl;
      }else {
        std::make_shared<Session>(std::move(socket))->start();
      }
      doAccept();
    });
  }
};


int main() {
  std::thread reader(read_thread);

  asio::io_service io_service;

  Server server(io_service, 12346);
  server.doAccept();

  io_service.run();
  reader.join();
}