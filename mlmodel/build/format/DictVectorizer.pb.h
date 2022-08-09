// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: DictVectorizer.proto

#ifndef PROTOBUF_DictVectorizer_2eproto__INCLUDED
#define PROTOBUF_DictVectorizer_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3003000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3003000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include "DataStructures.pb.h"  // IWYU pragma: export
// @@protoc_insertion_point(includes)
namespace CoreML {
namespace Specification {
class ArrayFeatureType;
class ArrayFeatureTypeDefaultTypeInternal;
extern ArrayFeatureTypeDefaultTypeInternal _ArrayFeatureType_default_instance_;
class ArrayFeatureType_EnumeratedShapes;
class ArrayFeatureType_EnumeratedShapesDefaultTypeInternal;
extern ArrayFeatureType_EnumeratedShapesDefaultTypeInternal _ArrayFeatureType_EnumeratedShapes_default_instance_;
class ArrayFeatureType_Shape;
class ArrayFeatureType_ShapeDefaultTypeInternal;
extern ArrayFeatureType_ShapeDefaultTypeInternal _ArrayFeatureType_Shape_default_instance_;
class ArrayFeatureType_ShapeRange;
class ArrayFeatureType_ShapeRangeDefaultTypeInternal;
extern ArrayFeatureType_ShapeRangeDefaultTypeInternal _ArrayFeatureType_ShapeRange_default_instance_;
class DictVectorizer;
class DictVectorizerDefaultTypeInternal;
extern DictVectorizerDefaultTypeInternal _DictVectorizer_default_instance_;
class DictionaryFeatureType;
class DictionaryFeatureTypeDefaultTypeInternal;
extern DictionaryFeatureTypeDefaultTypeInternal _DictionaryFeatureType_default_instance_;
class DoubleFeatureType;
class DoubleFeatureTypeDefaultTypeInternal;
extern DoubleFeatureTypeDefaultTypeInternal _DoubleFeatureType_default_instance_;
class DoubleRange;
class DoubleRangeDefaultTypeInternal;
extern DoubleRangeDefaultTypeInternal _DoubleRange_default_instance_;
class DoubleVector;
class DoubleVectorDefaultTypeInternal;
extern DoubleVectorDefaultTypeInternal _DoubleVector_default_instance_;
class FeatureType;
class FeatureTypeDefaultTypeInternal;
extern FeatureTypeDefaultTypeInternal _FeatureType_default_instance_;
class FloatVector;
class FloatVectorDefaultTypeInternal;
extern FloatVectorDefaultTypeInternal _FloatVector_default_instance_;
class ImageFeatureType;
class ImageFeatureTypeDefaultTypeInternal;
extern ImageFeatureTypeDefaultTypeInternal _ImageFeatureType_default_instance_;
class ImageFeatureType_EnumeratedImageSizes;
class ImageFeatureType_EnumeratedImageSizesDefaultTypeInternal;
extern ImageFeatureType_EnumeratedImageSizesDefaultTypeInternal _ImageFeatureType_EnumeratedImageSizes_default_instance_;
class ImageFeatureType_ImageSize;
class ImageFeatureType_ImageSizeDefaultTypeInternal;
extern ImageFeatureType_ImageSizeDefaultTypeInternal _ImageFeatureType_ImageSize_default_instance_;
class ImageFeatureType_ImageSizeRange;
class ImageFeatureType_ImageSizeRangeDefaultTypeInternal;
extern ImageFeatureType_ImageSizeRangeDefaultTypeInternal _ImageFeatureType_ImageSizeRange_default_instance_;
class Int64FeatureType;
class Int64FeatureTypeDefaultTypeInternal;
extern Int64FeatureTypeDefaultTypeInternal _Int64FeatureType_default_instance_;
class Int64Range;
class Int64RangeDefaultTypeInternal;
extern Int64RangeDefaultTypeInternal _Int64Range_default_instance_;
class Int64Set;
class Int64SetDefaultTypeInternal;
extern Int64SetDefaultTypeInternal _Int64Set_default_instance_;
class Int64ToDoubleMap;
class Int64ToDoubleMapDefaultTypeInternal;
extern Int64ToDoubleMapDefaultTypeInternal _Int64ToDoubleMap_default_instance_;
class Int64ToDoubleMap_MapEntry;
class Int64ToDoubleMap_MapEntryDefaultTypeInternal;
extern Int64ToDoubleMap_MapEntryDefaultTypeInternal _Int64ToDoubleMap_MapEntry_default_instance_;
class Int64ToStringMap;
class Int64ToStringMapDefaultTypeInternal;
extern Int64ToStringMapDefaultTypeInternal _Int64ToStringMap_default_instance_;
class Int64ToStringMap_MapEntry;
class Int64ToStringMap_MapEntryDefaultTypeInternal;
extern Int64ToStringMap_MapEntryDefaultTypeInternal _Int64ToStringMap_MapEntry_default_instance_;
class Int64Vector;
class Int64VectorDefaultTypeInternal;
extern Int64VectorDefaultTypeInternal _Int64Vector_default_instance_;
class SequenceFeatureType;
class SequenceFeatureTypeDefaultTypeInternal;
extern SequenceFeatureTypeDefaultTypeInternal _SequenceFeatureType_default_instance_;
class SizeRange;
class SizeRangeDefaultTypeInternal;
extern SizeRangeDefaultTypeInternal _SizeRange_default_instance_;
class StringFeatureType;
class StringFeatureTypeDefaultTypeInternal;
extern StringFeatureTypeDefaultTypeInternal _StringFeatureType_default_instance_;
class StringToDoubleMap;
class StringToDoubleMapDefaultTypeInternal;
extern StringToDoubleMapDefaultTypeInternal _StringToDoubleMap_default_instance_;
class StringToDoubleMap_MapEntry;
class StringToDoubleMap_MapEntryDefaultTypeInternal;
extern StringToDoubleMap_MapEntryDefaultTypeInternal _StringToDoubleMap_MapEntry_default_instance_;
class StringToInt64Map;
class StringToInt64MapDefaultTypeInternal;
extern StringToInt64MapDefaultTypeInternal _StringToInt64Map_default_instance_;
class StringToInt64Map_MapEntry;
class StringToInt64Map_MapEntryDefaultTypeInternal;
extern StringToInt64Map_MapEntryDefaultTypeInternal _StringToInt64Map_MapEntry_default_instance_;
class StringVector;
class StringVectorDefaultTypeInternal;
extern StringVectorDefaultTypeInternal _StringVector_default_instance_;
}  // namespace Specification
}  // namespace CoreML

namespace CoreML {
namespace Specification {

namespace protobuf_DictVectorizer_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_DictVectorizer_2eproto

// ===================================================================

class DictVectorizer : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.DictVectorizer) */ {
 public:
  DictVectorizer();
  virtual ~DictVectorizer();

  DictVectorizer(const DictVectorizer& from);

  inline DictVectorizer& operator=(const DictVectorizer& from) {
    CopyFrom(from);
    return *this;
  }

  static const DictVectorizer& default_instance();

  enum MapCase {
    kStringToIndex = 1,
    kInt64ToIndex = 2,
    MAP_NOT_SET = 0,
  };

  static inline const DictVectorizer* internal_default_instance() {
    return reinterpret_cast<const DictVectorizer*>(
               &_DictVectorizer_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(DictVectorizer* other);

  // implements Message ----------------------------------------------

  inline DictVectorizer* New() const PROTOBUF_FINAL { return New(NULL); }

  DictVectorizer* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const DictVectorizer& from);
  void MergeFrom(const DictVectorizer& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  void DiscardUnknownFields();
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(DictVectorizer* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::std::string GetTypeName() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .CoreML.Specification.StringVector stringToIndex = 1;
  bool has_stringtoindex() const;
  void clear_stringtoindex();
  static const int kStringToIndexFieldNumber = 1;
  const ::CoreML::Specification::StringVector& stringtoindex() const;
  ::CoreML::Specification::StringVector* mutable_stringtoindex();
  ::CoreML::Specification::StringVector* release_stringtoindex();
  void set_allocated_stringtoindex(::CoreML::Specification::StringVector* stringtoindex);

  // .CoreML.Specification.Int64Vector int64ToIndex = 2;
  bool has_int64toindex() const;
  void clear_int64toindex();
  static const int kInt64ToIndexFieldNumber = 2;
  const ::CoreML::Specification::Int64Vector& int64toindex() const;
  ::CoreML::Specification::Int64Vector* mutable_int64toindex();
  ::CoreML::Specification::Int64Vector* release_int64toindex();
  void set_allocated_int64toindex(::CoreML::Specification::Int64Vector* int64toindex);

  MapCase Map_case() const;
  // @@protoc_insertion_point(class_scope:CoreML.Specification.DictVectorizer)
 private:
  void set_has_stringtoindex();
  void set_has_int64toindex();

  inline bool has_Map() const;
  void clear_Map();
  inline void clear_has_Map();

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  union MapUnion {
    MapUnion() {}
    ::CoreML::Specification::StringVector* stringtoindex_;
    ::CoreML::Specification::Int64Vector* int64toindex_;
  } Map_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct protobuf_DictVectorizer_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// DictVectorizer

// .CoreML.Specification.StringVector stringToIndex = 1;
inline bool DictVectorizer::has_stringtoindex() const {
  return Map_case() == kStringToIndex;
}
inline void DictVectorizer::set_has_stringtoindex() {
  _oneof_case_[0] = kStringToIndex;
}
inline void DictVectorizer::clear_stringtoindex() {
  if (has_stringtoindex()) {
    delete Map_.stringtoindex_;
    clear_has_Map();
  }
}
inline  const ::CoreML::Specification::StringVector& DictVectorizer::stringtoindex() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.DictVectorizer.stringToIndex)
  return has_stringtoindex()
      ? *Map_.stringtoindex_
      : ::CoreML::Specification::StringVector::default_instance();
}
inline ::CoreML::Specification::StringVector* DictVectorizer::mutable_stringtoindex() {
  if (!has_stringtoindex()) {
    clear_Map();
    set_has_stringtoindex();
    Map_.stringtoindex_ = new ::CoreML::Specification::StringVector;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.DictVectorizer.stringToIndex)
  return Map_.stringtoindex_;
}
inline ::CoreML::Specification::StringVector* DictVectorizer::release_stringtoindex() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.DictVectorizer.stringToIndex)
  if (has_stringtoindex()) {
    clear_has_Map();
    ::CoreML::Specification::StringVector* temp = Map_.stringtoindex_;
    Map_.stringtoindex_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
inline void DictVectorizer::set_allocated_stringtoindex(::CoreML::Specification::StringVector* stringtoindex) {
  clear_Map();
  if (stringtoindex) {
    set_has_stringtoindex();
    Map_.stringtoindex_ = stringtoindex;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.DictVectorizer.stringToIndex)
}

// .CoreML.Specification.Int64Vector int64ToIndex = 2;
inline bool DictVectorizer::has_int64toindex() const {
  return Map_case() == kInt64ToIndex;
}
inline void DictVectorizer::set_has_int64toindex() {
  _oneof_case_[0] = kInt64ToIndex;
}
inline void DictVectorizer::clear_int64toindex() {
  if (has_int64toindex()) {
    delete Map_.int64toindex_;
    clear_has_Map();
  }
}
inline  const ::CoreML::Specification::Int64Vector& DictVectorizer::int64toindex() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.DictVectorizer.int64ToIndex)
  return has_int64toindex()
      ? *Map_.int64toindex_
      : ::CoreML::Specification::Int64Vector::default_instance();
}
inline ::CoreML::Specification::Int64Vector* DictVectorizer::mutable_int64toindex() {
  if (!has_int64toindex()) {
    clear_Map();
    set_has_int64toindex();
    Map_.int64toindex_ = new ::CoreML::Specification::Int64Vector;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.DictVectorizer.int64ToIndex)
  return Map_.int64toindex_;
}
inline ::CoreML::Specification::Int64Vector* DictVectorizer::release_int64toindex() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.DictVectorizer.int64ToIndex)
  if (has_int64toindex()) {
    clear_has_Map();
    ::CoreML::Specification::Int64Vector* temp = Map_.int64toindex_;
    Map_.int64toindex_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
inline void DictVectorizer::set_allocated_int64toindex(::CoreML::Specification::Int64Vector* int64toindex) {
  clear_Map();
  if (int64toindex) {
    set_has_int64toindex();
    Map_.int64toindex_ = int64toindex;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.DictVectorizer.int64ToIndex)
}

inline bool DictVectorizer::has_Map() const {
  return Map_case() != MAP_NOT_SET;
}
inline void DictVectorizer::clear_has_Map() {
  _oneof_case_[0] = MAP_NOT_SET;
}
inline DictVectorizer::MapCase DictVectorizer::Map_case() const {
  return DictVectorizer::MapCase(_oneof_case_[0]);
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_DictVectorizer_2eproto__INCLUDED