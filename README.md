# MNIST Dataset Handler

## Overview
C++17 library for handling MNIST-like datasets.
The library provides template-based data handlers that support different container types and value types.

## Features
- IDX file format support (both images and labels)
- Template-based containers (std::vector and std::array support)
- RAII-compliant memory management
- Integrated logging system using glog
- Shape information and data validation

## Requirements
- C++17 compliant compiler
- CMake 3.28.3 or higher
- vcpkg package manager
- Dependencies are managed through vcpkg manifest:
  - glog
  - gflags
  - Boost::system

## Project Structure
```
.
├── CMakeLists.txt          # Main CMake configuration
├── mlcpp/                  # Library source
│   └── mnist_etl/          # MNIST ETL functionality
│       ├── datahandler.hpp # Main data handling class
│       └── helper_typetraits.hpp
├── executables/            # Example executables
└── data/                   # MNIST dataset files
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## Building

### Basic Build
The project includes a build script (`build.sh`) that handles the build process:

```bash
./build.sh
```

The script performs the following operations:
1. Cleans the build directory (`out/build/*`)
2. Configures CMake with vcpkg toolchain
3. Builds the project

### Build Options

- Debug build:
```bash
cmake -S . -B out/build/ -DCMAKE_TOOLCHAIN_FILE=[path_to_vcpkg]/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug
```

- Release build (default):
```bash
cmake -S . -B out/build/ -DCMAKE_TOOLCHAIN_FILE=[path_to_vcpkg]/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
```

- Parallel build (recommended):
```bash
cmake --build out/build/ -j$(nproc)
```

## Usage Example

```cpp
#include <datahandler.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <filesystem>

using namespace lm;

int main(int argc, char* argv[]) {
    // Initialize logging
    google::InitGoogleLogging(argv[0]);
    
    // Define types for images and labels
    using LabelValueType = uint8_t;
    using ImageContainer = std::array<uint8_t, 784u>;  // 28x28 MNIST images
    
    try {
        // Create data handler with MNIST dataset
        DataHandler<ImageContainer, LabelValueType> handler(
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte"
        );
        
        // Print dataset information
        handler.PrintShape();
        
        // Get a single instance (image and label)
        auto [image, label] = handler.GetDatasetInstance(0);
        
    } catch(const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
```

## API Documentation

### DataHandler Class
The main class for handling MNIST dataset operations.

```cpp
template<typename ImageContainer, typename LabelValueType>
class DataHandler;
```

#### Constructor
```cpp
DataHandler(const std::string& images_path, const std::string& labels_path);
```
Creates a data handler instance by loading images and labels from the specified paths.

#### Methods
- `const DataSetType& Data() const`
  Returns a const reference to the dataset.

- `void PrintShape() const`
  Prints the shape information of the dataset.

- `std::pair<ImageContainer, LabelValueType> GetDatasetInstance(std::size_t index) const`
  Retrieves a specific instance (image and label) from the dataset.

### Supported Types
- ImageContainer: `std::vector` or `std::array`
- LabelValueType: Usually `uint8_t` for MNIST labels

### Common Build Issues
- Ensure VCPKG_ROOT environment variable is set
- Verify vcpkg is properly installed
- Check CMAKE_TOOLCHAIN_FILE path
- Verify vcpkg manifest mode is enabled

## Acknowledgments
- MNIST dataset creators
- Contributors and maintainers
