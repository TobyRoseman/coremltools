// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: WordTagger.proto

#ifndef PROTOBUF_WordTagger_2eproto__INCLUDED
#define PROTOBUF_WordTagger_2eproto__INCLUDED

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
class PrecisionRecallCurve;
class PrecisionRecallCurveDefaultTypeInternal;
extern PrecisionRecallCurveDefaultTypeInternal _PrecisionRecallCurve_default_instance_;
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
namespace CoreMLModels {
class WordTagger;
class WordTaggerDefaultTypeInternal;
extern WordTaggerDefaultTypeInternal _WordTagger_default_instance_;
}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML

namespace CoreML {
namespace Specification {
namespace CoreMLModels {

namespace protobuf_WordTagger_2eproto {
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
}  // namespace protobuf_WordTagger_2eproto

// ===================================================================

class WordTagger : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.CoreMLModels.WordTagger) */ {
 public:
  WordTagger();
  virtual ~WordTagger();

  WordTagger(const WordTagger& from);

  inline WordTagger& operator=(const WordTagger& from) {
    CopyFrom(from);
    return *this;
  }

  static const WordTagger& default_instance();

  enum TagsCase {
    kStringTags = 200,
    TAGS_NOT_SET = 0,
  };

  static inline const WordTagger* internal_default_instance() {
    return reinterpret_cast<const WordTagger*>(
               &_WordTagger_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(WordTagger* other);

  // implements Message ----------------------------------------------

  inline WordTagger* New() const PROTOBUF_FINAL { return New(NULL); }

  WordTagger* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from)
    PROTOBUF_FINAL;
  void CopyFrom(const WordTagger& from);
  void MergeFrom(const WordTagger& from);
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
  void InternalSwap(WordTagger* other);
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

  // string language = 10;
  void clear_language();
  static const int kLanguageFieldNumber = 10;
  const ::std::string& language() const;
  void set_language(const ::std::string& value);
  #if LANG_CXX11
  void set_language(::std::string&& value);
  #endif
  void set_language(const char* value);
  void set_language(const char* value, size_t size);
  ::std::string* mutable_language();
  ::std::string* release_language();
  void set_allocated_language(::std::string* language);

  // string tokensOutputFeatureName = 20;
  void clear_tokensoutputfeaturename();
  static const int kTokensOutputFeatureNameFieldNumber = 20;
  const ::std::string& tokensoutputfeaturename() const;
  void set_tokensoutputfeaturename(const ::std::string& value);
  #if LANG_CXX11
  void set_tokensoutputfeaturename(::std::string&& value);
  #endif
  void set_tokensoutputfeaturename(const char* value);
  void set_tokensoutputfeaturename(const char* value, size_t size);
  ::std::string* mutable_tokensoutputfeaturename();
  ::std::string* release_tokensoutputfeaturename();
  void set_allocated_tokensoutputfeaturename(::std::string* tokensoutputfeaturename);

  // string tokenTagsOutputFeatureName = 21;
  void clear_tokentagsoutputfeaturename();
  static const int kTokenTagsOutputFeatureNameFieldNumber = 21;
  const ::std::string& tokentagsoutputfeaturename() const;
  void set_tokentagsoutputfeaturename(const ::std::string& value);
  #if LANG_CXX11
  void set_tokentagsoutputfeaturename(::std::string&& value);
  #endif
  void set_tokentagsoutputfeaturename(const char* value);
  void set_tokentagsoutputfeaturename(const char* value, size_t size);
  ::std::string* mutable_tokentagsoutputfeaturename();
  ::std::string* release_tokentagsoutputfeaturename();
  void set_allocated_tokentagsoutputfeaturename(::std::string* tokentagsoutputfeaturename);

  // string tokenLocationsOutputFeatureName = 22;
  void clear_tokenlocationsoutputfeaturename();
  static const int kTokenLocationsOutputFeatureNameFieldNumber = 22;
  const ::std::string& tokenlocationsoutputfeaturename() const;
  void set_tokenlocationsoutputfeaturename(const ::std::string& value);
  #if LANG_CXX11
  void set_tokenlocationsoutputfeaturename(::std::string&& value);
  #endif
  void set_tokenlocationsoutputfeaturename(const char* value);
  void set_tokenlocationsoutputfeaturename(const char* value, size_t size);
  ::std::string* mutable_tokenlocationsoutputfeaturename();
  ::std::string* release_tokenlocationsoutputfeaturename();
  void set_allocated_tokenlocationsoutputfeaturename(::std::string* tokenlocationsoutputfeaturename);

  // string tokenLengthsOutputFeatureName = 23;
  void clear_tokenlengthsoutputfeaturename();
  static const int kTokenLengthsOutputFeatureNameFieldNumber = 23;
  const ::std::string& tokenlengthsoutputfeaturename() const;
  void set_tokenlengthsoutputfeaturename(const ::std::string& value);
  #if LANG_CXX11
  void set_tokenlengthsoutputfeaturename(::std::string&& value);
  #endif
  void set_tokenlengthsoutputfeaturename(const char* value);
  void set_tokenlengthsoutputfeaturename(const char* value, size_t size);
  ::std::string* mutable_tokenlengthsoutputfeaturename();
  ::std::string* release_tokenlengthsoutputfeaturename();
  void set_allocated_tokenlengthsoutputfeaturename(::std::string* tokenlengthsoutputfeaturename);

  // bytes modelParameterData = 100;
  void clear_modelparameterdata();
  static const int kModelParameterDataFieldNumber = 100;
  const ::std::string& modelparameterdata() const;
  void set_modelparameterdata(const ::std::string& value);
  #if LANG_CXX11
  void set_modelparameterdata(::std::string&& value);
  #endif
  void set_modelparameterdata(const char* value);
  void set_modelparameterdata(const void* value, size_t size);
  ::std::string* mutable_modelparameterdata();
  ::std::string* release_modelparameterdata();
  void set_allocated_modelparameterdata(::std::string* modelparameterdata);

  // uint32 revision = 1;
  void clear_revision();
  static const int kRevisionFieldNumber = 1;
  ::google::protobuf::uint32 revision() const;
  void set_revision(::google::protobuf::uint32 value);

  // .CoreML.Specification.StringVector stringTags = 200;
  bool has_stringtags() const;
  void clear_stringtags();
  static const int kStringTagsFieldNumber = 200;
  const ::CoreML::Specification::StringVector& stringtags() const;
  ::CoreML::Specification::StringVector* mutable_stringtags();
  ::CoreML::Specification::StringVector* release_stringtags();
  void set_allocated_stringtags(::CoreML::Specification::StringVector* stringtags);

  TagsCase Tags_case() const;
  // @@protoc_insertion_point(class_scope:CoreML.Specification.CoreMLModels.WordTagger)
 private:
  void set_has_stringtags();

  inline bool has_Tags() const;
  void clear_Tags();
  inline void clear_has_Tags();

  ::google::protobuf::internal::InternalMetadataWithArenaLite _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr language_;
  ::google::protobuf::internal::ArenaStringPtr tokensoutputfeaturename_;
  ::google::protobuf::internal::ArenaStringPtr tokentagsoutputfeaturename_;
  ::google::protobuf::internal::ArenaStringPtr tokenlocationsoutputfeaturename_;
  ::google::protobuf::internal::ArenaStringPtr tokenlengthsoutputfeaturename_;
  ::google::protobuf::internal::ArenaStringPtr modelparameterdata_;
  ::google::protobuf::uint32 revision_;
  union TagsUnion {
    TagsUnion() {}
    ::CoreML::Specification::StringVector* stringtags_;
  } Tags_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct protobuf_WordTagger_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// WordTagger

// uint32 revision = 1;
inline void WordTagger::clear_revision() {
  revision_ = 0u;
}
inline ::google::protobuf::uint32 WordTagger::revision() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.revision)
  return revision_;
}
inline void WordTagger::set_revision(::google::protobuf::uint32 value) {
  
  revision_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.revision)
}

// string language = 10;
inline void WordTagger::clear_language() {
  language_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::language() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.language)
  return language_.GetNoArena();
}
inline void WordTagger::set_language(const ::std::string& value) {
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.language)
}
#if LANG_CXX11
inline void WordTagger::set_language(::std::string&& value) {
  
  language_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.language)
}
#endif
inline void WordTagger::set_language(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.language)
}
inline void WordTagger::set_language(const char* value, size_t size) {
  
  language_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.language)
}
inline ::std::string* WordTagger::mutable_language() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.language)
  return language_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_language() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.language)
  
  return language_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_language(::std::string* language) {
  if (language != NULL) {
    
  } else {
    
  }
  language_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), language);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.language)
}

// string tokensOutputFeatureName = 20;
inline void WordTagger::clear_tokensoutputfeaturename() {
  tokensoutputfeaturename_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::tokensoutputfeaturename() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
  return tokensoutputfeaturename_.GetNoArena();
}
inline void WordTagger::set_tokensoutputfeaturename(const ::std::string& value) {
  
  tokensoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
}
#if LANG_CXX11
inline void WordTagger::set_tokensoutputfeaturename(::std::string&& value) {
  
  tokensoutputfeaturename_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
}
#endif
inline void WordTagger::set_tokensoutputfeaturename(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  tokensoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
}
inline void WordTagger::set_tokensoutputfeaturename(const char* value, size_t size) {
  
  tokensoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
}
inline ::std::string* WordTagger::mutable_tokensoutputfeaturename() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
  return tokensoutputfeaturename_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_tokensoutputfeaturename() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
  
  return tokensoutputfeaturename_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_tokensoutputfeaturename(::std::string* tokensoutputfeaturename) {
  if (tokensoutputfeaturename != NULL) {
    
  } else {
    
  }
  tokensoutputfeaturename_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), tokensoutputfeaturename);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.tokensOutputFeatureName)
}

// string tokenTagsOutputFeatureName = 21;
inline void WordTagger::clear_tokentagsoutputfeaturename() {
  tokentagsoutputfeaturename_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::tokentagsoutputfeaturename() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
  return tokentagsoutputfeaturename_.GetNoArena();
}
inline void WordTagger::set_tokentagsoutputfeaturename(const ::std::string& value) {
  
  tokentagsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
}
#if LANG_CXX11
inline void WordTagger::set_tokentagsoutputfeaturename(::std::string&& value) {
  
  tokentagsoutputfeaturename_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
}
#endif
inline void WordTagger::set_tokentagsoutputfeaturename(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  tokentagsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
}
inline void WordTagger::set_tokentagsoutputfeaturename(const char* value, size_t size) {
  
  tokentagsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
}
inline ::std::string* WordTagger::mutable_tokentagsoutputfeaturename() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
  return tokentagsoutputfeaturename_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_tokentagsoutputfeaturename() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
  
  return tokentagsoutputfeaturename_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_tokentagsoutputfeaturename(::std::string* tokentagsoutputfeaturename) {
  if (tokentagsoutputfeaturename != NULL) {
    
  } else {
    
  }
  tokentagsoutputfeaturename_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), tokentagsoutputfeaturename);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.tokenTagsOutputFeatureName)
}

// string tokenLocationsOutputFeatureName = 22;
inline void WordTagger::clear_tokenlocationsoutputfeaturename() {
  tokenlocationsoutputfeaturename_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::tokenlocationsoutputfeaturename() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
  return tokenlocationsoutputfeaturename_.GetNoArena();
}
inline void WordTagger::set_tokenlocationsoutputfeaturename(const ::std::string& value) {
  
  tokenlocationsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
}
#if LANG_CXX11
inline void WordTagger::set_tokenlocationsoutputfeaturename(::std::string&& value) {
  
  tokenlocationsoutputfeaturename_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
}
#endif
inline void WordTagger::set_tokenlocationsoutputfeaturename(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  tokenlocationsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
}
inline void WordTagger::set_tokenlocationsoutputfeaturename(const char* value, size_t size) {
  
  tokenlocationsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
}
inline ::std::string* WordTagger::mutable_tokenlocationsoutputfeaturename() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
  return tokenlocationsoutputfeaturename_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_tokenlocationsoutputfeaturename() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
  
  return tokenlocationsoutputfeaturename_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_tokenlocationsoutputfeaturename(::std::string* tokenlocationsoutputfeaturename) {
  if (tokenlocationsoutputfeaturename != NULL) {
    
  } else {
    
  }
  tokenlocationsoutputfeaturename_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), tokenlocationsoutputfeaturename);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.tokenLocationsOutputFeatureName)
}

// string tokenLengthsOutputFeatureName = 23;
inline void WordTagger::clear_tokenlengthsoutputfeaturename() {
  tokenlengthsoutputfeaturename_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::tokenlengthsoutputfeaturename() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
  return tokenlengthsoutputfeaturename_.GetNoArena();
}
inline void WordTagger::set_tokenlengthsoutputfeaturename(const ::std::string& value) {
  
  tokenlengthsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
}
#if LANG_CXX11
inline void WordTagger::set_tokenlengthsoutputfeaturename(::std::string&& value) {
  
  tokenlengthsoutputfeaturename_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
}
#endif
inline void WordTagger::set_tokenlengthsoutputfeaturename(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  tokenlengthsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
}
inline void WordTagger::set_tokenlengthsoutputfeaturename(const char* value, size_t size) {
  
  tokenlengthsoutputfeaturename_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
}
inline ::std::string* WordTagger::mutable_tokenlengthsoutputfeaturename() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
  return tokenlengthsoutputfeaturename_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_tokenlengthsoutputfeaturename() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
  
  return tokenlengthsoutputfeaturename_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_tokenlengthsoutputfeaturename(::std::string* tokenlengthsoutputfeaturename) {
  if (tokenlengthsoutputfeaturename != NULL) {
    
  } else {
    
  }
  tokenlengthsoutputfeaturename_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), tokenlengthsoutputfeaturename);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.tokenLengthsOutputFeatureName)
}

// bytes modelParameterData = 100;
inline void WordTagger::clear_modelparameterdata() {
  modelparameterdata_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& WordTagger::modelparameterdata() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
  return modelparameterdata_.GetNoArena();
}
inline void WordTagger::set_modelparameterdata(const ::std::string& value) {
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
}
#if LANG_CXX11
inline void WordTagger::set_modelparameterdata(::std::string&& value) {
  
  modelparameterdata_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
}
#endif
inline void WordTagger::set_modelparameterdata(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
}
inline void WordTagger::set_modelparameterdata(const void* value, size_t size) {
  
  modelparameterdata_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
}
inline ::std::string* WordTagger::mutable_modelparameterdata() {
  
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
  return modelparameterdata_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* WordTagger::release_modelparameterdata() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
  
  return modelparameterdata_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void WordTagger::set_allocated_modelparameterdata(::std::string* modelparameterdata) {
  if (modelparameterdata != NULL) {
    
  } else {
    
  }
  modelparameterdata_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), modelparameterdata);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.modelParameterData)
}

// .CoreML.Specification.StringVector stringTags = 200;
inline bool WordTagger::has_stringtags() const {
  return Tags_case() == kStringTags;
}
inline void WordTagger::set_has_stringtags() {
  _oneof_case_[0] = kStringTags;
}
inline void WordTagger::clear_stringtags() {
  if (has_stringtags()) {
    delete Tags_.stringtags_;
    clear_has_Tags();
  }
}
inline  const ::CoreML::Specification::StringVector& WordTagger::stringtags() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.CoreMLModels.WordTagger.stringTags)
  return has_stringtags()
      ? *Tags_.stringtags_
      : ::CoreML::Specification::StringVector::default_instance();
}
inline ::CoreML::Specification::StringVector* WordTagger::mutable_stringtags() {
  if (!has_stringtags()) {
    clear_Tags();
    set_has_stringtags();
    Tags_.stringtags_ = new ::CoreML::Specification::StringVector;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.CoreMLModels.WordTagger.stringTags)
  return Tags_.stringtags_;
}
inline ::CoreML::Specification::StringVector* WordTagger::release_stringtags() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.CoreMLModels.WordTagger.stringTags)
  if (has_stringtags()) {
    clear_has_Tags();
    ::CoreML::Specification::StringVector* temp = Tags_.stringtags_;
    Tags_.stringtags_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
inline void WordTagger::set_allocated_stringtags(::CoreML::Specification::StringVector* stringtags) {
  clear_Tags();
  if (stringtags) {
    set_has_stringtags();
    Tags_.stringtags_ = stringtags;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.CoreMLModels.WordTagger.stringTags)
}

inline bool WordTagger::has_Tags() const {
  return Tags_case() != TAGS_NOT_SET;
}
inline void WordTagger::clear_has_Tags() {
  _oneof_case_[0] = TAGS_NOT_SET;
}
inline WordTagger::TagsCase WordTagger::Tags_case() const {
  return WordTagger::TagsCase(_oneof_case_[0]);
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


}  // namespace CoreMLModels
}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_WordTagger_2eproto__INCLUDED
