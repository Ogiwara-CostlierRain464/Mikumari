#include <iostream>
#include <boost/asio.hpp>
#include <boost/version.hpp>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <chrono>

namespace asio = boost::asio;
using asio::ip::tcp;
using namespace std::chrono_literals;

tbb::concurrent_bounded_queue<bool> queue{};

void read_thread() {
  for(;;) {
    bool req;
    queue.pop(req);
    std::cout << "new req" << std::endl;
  }
}

class Session : public std::enable_shared_from_this<Session>{
  tcp::socket socket;

public:
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
      std::cout << "request to use " << data << " model" << std::endl;

      //queue.push(false);

      // connect to worker and wait
      asio::io_service io_service;
      tcp::socket w_socket(io_service);

      w_socket.connect(tcp::endpoint(
        asio::ip::address::from_string("127.0.0.1"), 12346
        ));

      const std::string msg_to_worker = data;
      asio::write(w_socket, asio::buffer(msg_to_worker), err);
      assert(!err);

      asio::streambuf buf2;
      asio::read_until(w_socket, buf2, "\0",err);
      assert(!err);

      std::cout << "worker replied!" << std::endl;

      asio::write(socket, buf2, err);

      socket.close();
    }
  }
};


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

  Server server(io_service, 12345);
  server.doAccept();

  io_service.run();
  reader.join();
}