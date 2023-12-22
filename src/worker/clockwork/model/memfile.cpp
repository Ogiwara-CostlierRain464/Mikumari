#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include "dmlc/logging.h"
#include "memfile.h"
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <sstream>
#include <cstdio>

namespace clockwork {

int make_shmemfd(std::string &name) {
	int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, S_IRWXU);
	if (fd < 0) {
		std::cout << fd << std::endl;
		perror("shm_open");
		CHECK(fd < 0) << "Could not create memfd using shm_open";
	}
	return fd;
}

int make_memfd() {
	int fd = syscall(__NR_memfd_create, std::string("").c_str(), 0);
	if (fd < 0) {
		std::cout << fd << std::endl;
		perror("memfd_create");
		CHECK(fd < 0) << "Could not create memfd using memfd_create";
	}
	return fd;
}

std::string memfd_filename(int memfd) {
	std::stringstream ss;
	ss << "/proc/" << getpid() << "/fd/" << memfd;
	return ss.str();
}

MemfdFile* MemfdFile::readFrom(const std::string &filename) {
	int memfd = make_memfd();
	std::string memfilename = memfd_filename(memfd);
    std::ofstream dst(memfilename, std::ios::binary);
    CHECK(dst.good()) << "Bad memfile " << memfilename;

	std::ifstream src(filename, std::ios::binary);
	CHECK(src.good()) << "Unable to open file " << filename;
    dst << src.rdbuf();

    src.close();
    dst.close();

    return new MemfdFile(memfd, memfilename);
}

int MemfdFile::close() {
	std::cout << "Closing " << memfd << std::endl;
	return ::close(memfd);
}

unsigned shmem_counter = 0;

ShmemFile* ShmemFile::readFrom(const std::string &filename) {
	// Filename of the shmem file
	std::stringstream name;
	name << "/clockwork-" << getpid() << "-" << shmem_counter++;
	std::string shmem_name = name.str();
	int shmem_fd = make_shmemfd(shmem_name);

	// Path to the shmem file
	std::string shmem_path = "/dev/shm" + shmem_name;

	// Remove existing file
	std::remove(shmem_path.c_str());

    std::ofstream dst(shmem_path, std::ios::binary);
    CHECK(dst.good()) << "Bad memfile " << shmem_path;

	std::ifstream src(filename, std::ios::binary);
	CHECK(src.good()) << "Unable to open file " << filename;
    dst << src.rdbuf();

    src.close();
    dst.close();

    return new ShmemFile(shmem_fd, shmem_path, shmem_name);
}

int ShmemFile::close() {
	::close(memfd);
	int status = shm_unlink(name.c_str());
	return status;
}

}