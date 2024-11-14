#include <vector>
#include <array>
#include <type_traits>

// IS_ARRAY
template<typename T>
struct is_array : std::false_type{};

template<typename T, std::size_t N>
struct is_array<std::array<T, N> > : std::true_type{};

template<typename T>
constexpr bool is_array_v = is_array<T>::value; 

// IS_VECTOR
template<typename T>
struct is_vector: std::false_type{};

template<typename T>
struct is_vector<std::vector<T> > : std::true_type{};

template<typename T>
constexpr bool is_vector_v = is_vector<T>::value; 

// UNSUPPORTED TYPE FOR ALL OTHER TYPES CONTAINERS
template<typename T>
constexpr bool always_false_v = false;

// https://en.cppreference.com/w/cpp/container/array/tuple_size
template<typename T>
constexpr std::size_t tuple_size_v = std::tuple_size<T>::value;
