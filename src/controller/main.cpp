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
      std::cout << "receive: " << data << std::endl;

      queue.push(false);

      // wait worker
      std::this_thread::sleep_for(4s);
      std::cout << "sleep finished" << std::endl;

      std::string msg = "kick back!";
      asio::write(socket, asio::buffer(msg), err);

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