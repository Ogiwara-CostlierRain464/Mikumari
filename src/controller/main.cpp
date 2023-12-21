#include <iostream>
#include <boost/asio.hpp>
#include <boost/version.hpp>

namespace asio = boost::asio;
using asio::ip::tcp;

int main() {
  std::cout << "Boost v: " << BOOST_LIB_VERSION << std::endl;

  asio::io_service io_service;

  tcp::acceptor acc(io_service, tcp::endpoint(
    tcp::v4(), 12345
  ));
  tcp::socket socket(io_service);

  acc.accept(socket);

  asio::streambuf receive_buffer;
  boost::system::error_code err;
  asio::read(socket, receive_buffer,
    asio::transfer_all(), err);

  if(err && err != asio::error::eof) {
    std::cout << "receive failed: " << err.message() << std::endl;
  }else {
    auto data = asio::buffer_cast<const char*>(receive_buffer.data());
    std::cout << data << std::endl;
  }
}