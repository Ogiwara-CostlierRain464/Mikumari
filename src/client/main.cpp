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

  const std::string msg = "Hello!";
  boost::system::error_code err;
  asio::write(socket, asio::buffer(msg), err);
  assert(!err);


  asio::streambuf buf;
  asio::read_until(socket, buf, "\0",err);
  assert(!err);
  auto data = asio::buffer_cast<const char*>(buf.data());
  std::cout << data << std::endl;


}
