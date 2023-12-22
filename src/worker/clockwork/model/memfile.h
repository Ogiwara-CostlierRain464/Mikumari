#ifndef _CLOCKWORK_MEMFILE_H_
#define _CLOCKWORK_MEMFILE_H_

#include <string>
#include <istream>
#include <fstream>

namespace clockwork {

  /** An in-memory file */
  class MemfileImpl {
  public:
    const int memfd;
    const std::string filename;

    MemfileImpl(const int &memfd, const std::string &filename) :
      memfd(memfd), filename(filename) {}

    virtual int close() = 0;
  };

  /** An in-memory file */
  class MemfdFile : public MemfileImpl {
  public:

    MemfdFile(const int &memfd, const std::string &filename) :
      MemfileImpl(memfd, filename) {}

    // Copy another file into a MemfdFile
    static MemfdFile* readFrom(const std::string &filename);

    virtual int close();
  };

  class ShmemFile : public MemfileImpl {
  public:
    const std::string name;

    ShmemFile(const int &fd, const std::string &filename, const std::string &name) :
      MemfileImpl(fd, filename), name(name) {}

    // Copy another file into a ShmemFile
    static ShmemFile* readFrom(const std::string &filename);

    virtual int close();

  };

  class Memfile {
  private:
    MemfileImpl* impl;

  public:
    const int memfd;
    const std::string filename;

    Memfile(MemfileImpl* impl) : impl(impl), memfd(impl->memfd), filename(impl->filename) {}

    // Copy another file into the default memfile impl
    static Memfile readFrom(const std::string &filename) {
      return Memfile(ShmemFile::readFrom(filename));
    }

    int close() {
      if (impl != nullptr) {
        return impl->close();
      } else {
        return -1;
      }
    }
  };

}

#endif
