#include "modeldef.h"
#include <dmlc/logging.h>

using namespace clockwork::model;


void ModelDef::ReadFrom(const std::string &data, ModelDef &def) {
  pods::InputBuffer in(data.data(), data.size());
  pods::BinaryDeserializer<decltype(in)> deserializer(in);
  pods::Error status = deserializer.load(def);
  CHECK(status == pods::Error::NoError) << "Cannot deserialize minmodel";
}

void PageMappedModelDef::ReadFrom(const std::string &data, PageMappedModelDef &def) {
  pods::InputBuffer in(data.data(), data.size());
  pods::BinaryDeserializer<decltype(in)> deserializer(in);
  pods::Error status = deserializer.load(def);
  CHECK(status == pods::Error::NoError) << "Cannot deserialize minmodel";
}
