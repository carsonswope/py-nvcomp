#include <stdexcept>

// Convert numpy type string to nvcompType_t
nvcompType_t nvcomp_parse_np_type(const char* np_type_chars) {
    const std::string np_type_str(np_type_chars);
    if (np_type_str.compare("i4") == 0) {
        return nvcomp::TypeOf<int>();
    } else if (np_type_str.compare("u2") == 0) {
        return nvcomp::TypeOf<uint16_t>();
    } else {
        // TODO: more types!
        throw std::runtime_error("Unsupported type " + np_type_str);
    }
}
