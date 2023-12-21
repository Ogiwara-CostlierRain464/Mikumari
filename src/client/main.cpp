#include <iostream>
#include <boost/asio.hpp>

namespace asio = boost::asio;
using asio::ip::tcp;

int main() {
  asio::io_service io_service;
  tcp::socket socket(io_service);

  socket.connect(tcp::endpoint(
    asio::ip::address::from_string("127.0.0.1"), 12345
    ));

  const std::string msg = "a req from client";
  boost::system::error_code err;
  asio::write(socket, asio::buffer(msg), err);

  if(err) {
    std::cout << "send failed: " << err.message() << std::endl;
  }else {
    std::cout << "send correct" << std::endl;
  }
}
