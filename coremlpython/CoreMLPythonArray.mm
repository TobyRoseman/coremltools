#import "CoreMLPythonArray.h"

@implementation PybindCompatibleArray

+ (MLMultiArrayDataType)dataTypeOf:(py::array)array {
    const auto& dt = array.dtype();
    char kind = dt.kind();
    size_t itemsize = dt.itemsize();

    py::print("array.itemsize(): ", array.itemsize());
    
    if(kind == 'i') {
        return MLMultiArrayDataTypeInt32;
    } else if(kind == 'f') { 
        return MLMultiArrayDataTypeFloat32;
    } else if( (kind == 'f' || kind == 'd') ) { //&& itemsize == 8) {
        return MLMultiArrayDataTypeDouble;
    }

    py::print("kind: ", kind);
    
    throw std::runtime_error("Unsupported array type: " + std::to_string(kind) + " with itemsize = " + std::to_string(itemsize));
}

+ (NSArray<NSNumber *> *)shapeOf:(py::array)array {
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i < array.ndim(); i++) {
        [ret addObject:[NSNumber numberWithUnsignedLongLong:array.shape(i)]];
    }
    return ret;
}

+ (NSArray<NSNumber *> *)stridesOf:(py::array)array {
    // numpy strides is in bytes.
    // this type must return number of ELEMENTS! (as per mlkit)
    
    NSMutableArray<NSNumber *> *ret = [[NSMutableArray alloc] init];
    for (size_t i=0; i<array.ndim(); i++) {
        size_t stride = array.strides(i) / array.itemsize();
        [ret addObject:[NSNumber numberWithUnsignedLongLong:stride]];
    }
    return ret;
}

- (PybindCompatibleArray *)initWithArray:(py::array)array {

    py::print("===0===");
    py::print("array: ", array);
    py::print("array.ndim(): ", array.ndim());
    py::print("array.shape(0): ", array.shape(0));
    
    self = [super initWithDataPointer:array.mutable_data()
                                shape:[self.class shapeOf:array]
                             dataType:[self.class dataTypeOf:array]
                              strides:[self.class stridesOf:array]
                          deallocator:nil
                                error:nil];

    py::print("===1===");

    if (self) {
        m_array = array;
    }
    return self;
}

@end
