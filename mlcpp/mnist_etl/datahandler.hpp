#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

#include <vector>
#include <array>
#include <tuple>
#include <numeric>
#include <random>

#include "helper_typetraits.hpp"

#include <glog/logging.h>


namespace lm {


enum class FileType : uint32_t {
    IMAGE = 0x803,
    LABEL = 0x801
}; 


struct Header {
    FileType ftype; 
    uint32_t n_dim; // dim = number of dimension 
    std::vector<uint32_t> dim_sizes; 
}; 

/**
 * @brief Helper function to read IDX file header.
 * @return a struct header with the filetype, the number of dimensions in wich 
 * each instance of the dataset is represented and the size of each dimension in a vector of 
 * uint32.
 * 
 */
inline Header ReadHeader(const std::unique_ptr<char[]>& buff) {
    uint32_t* file_cursor = reinterpret_cast<uint32_t*>(buff.get()); 

    // big-endian to little-endian conversion 
    auto magic_num = __builtin_bswap32(*file_cursor); 
    
    FileType ftype_ = (magic_num == 0x803) ? FileType::IMAGE : FileType::LABEL; 
    uint32_t n_dim_ = magic_num & 0xFF; 

    std::vector<uint32_t> dim_sizes_; 

    dim_sizes_.reserve(n_dim_); 
    file_cursor++; 
    for (auto i=0u; i<n_dim_; i++) 
        dim_sizes_.push_back(__builtin_bswap32(*file_cursor++)); 

    return Header{ftype_, n_dim_, dim_sizes_}; 
}

/**
 * @brief Helper function to read IDX file data.
 * @return A tuple containing the file buffer, data size, and header information.
 * @todo Return a different tuple that will contain image_size or not according to the 
 * expected_filetype.
 * 
 */
inline auto ReadDataHelper(const std::string& path, FileType expected_filetype) {
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate); 
    if (!file) throw std::runtime_error("failed to open file from path: " + path); 

    auto file_size = file.tellg();
    auto buff = std::make_unique<char[]>(file_size); 
    file.seekg(0, std::ios::beg); 
    file.read(buff.get(), file_size);   

    if(file.gcount() != file_size)

    file.close();

    Header header = ReadHeader(buff);
    if (header.ftype != expected_filetype) {
        throw std::runtime_error(
            "Inconsistency between the filetype read from path and the expected filetype!"); 
    }

    auto image_size = std::accumulate(
    header.dim_sizes.begin()+1, header.dim_sizes.end(), 1u,
    [](uint32_t a, uint32_t b) {
        return a*b; 
    });

    return std::tuple<std::unique_ptr<char[]>, uint32_t, Header>{std::move(buff), image_size, header}; 
}


/**
 * @brief Reads image data from an IDX file into a vector of vectors.
 * @return A vector of vectors, where each inner vector represents a flattened image.
 */
template<typename T>
std::vector<std::vector<T> > ReadImages(const std::string& path){
    auto [buff, image_size, header] = ReadDataHelper(path, FileType::IMAGE); 
    auto buff_raw_ptr = reinterpret_cast<T*>(buff.get() + 4u*(header.n_dim + 1)); // skip header

    std::vector<std::vector<T> > images;
    images.reserve(header.dim_sizes[0]); 
    for(auto i=0u; i<header.dim_sizes[0]; i++){
        images.emplace_back(buff_raw_ptr, buff_raw_ptr + image_size); // flattened image
        buff_raw_ptr += image_size; 
    }

    return images; 
}

/**
 * @brief Reads image data from an IDX file into a vector of arrays.
 * @return A vector of arrays, where each array represents a flattened image.
 */
template<typename T, std::size_t N>
std::vector<std::array<T, N> > ReadImages(const std::string& path){
    auto [buff, image_size, header] = ReadDataHelper(path, FileType::IMAGE); 
    auto buff_raw_ptr = reinterpret_cast<T*>(buff.get() + 4u*(header.n_dim + 1)); // skip header

    std::vector<std::array<T, N> > images; 
    images.reserve(header.dim_sizes[0]); 
    for(auto i=0u; i<header.dim_sizes[0]; i++){
        std::array<T, N> img; 
        std::copy_n(buff_raw_ptr, image_size, img.begin()); // flattened image
        images.push_back(std::move(img)); 
        buff_raw_ptr += image_size; 
    }

    return images; 
}

/**
 * @brief Reads label data from an IDX file.
 * @return A vector of labels.
 */
template<typename LabelValueType>
std::vector<LabelValueType> ReadLabels(const std::string& path) {
    auto [buff, _, header] = ReadDataHelper(path, FileType::LABEL); 
    auto buff_raw_ptr = reinterpret_cast<LabelValueType*>(buff.get() + 4u*(header.n_dim + 1)); // skip header

    std::vector<LabelValueType> labels;
    labels.reserve(header.dim_sizes[0]); 
    for(auto i=0u; i<header.dim_sizes[0]; i++) {
        labels.push_back(*buff_raw_ptr);
        buff_raw_ptr++;
    }

    return labels;
}


/**
 * @brief A template structure to hold image data and corresponding labels.
 */
template<typename ImageContainer, typename LabelValueType>
struct Data {
    std::vector<ImageContainer> images; 
    std::vector<LabelValueType> labels; 
};

/**
 * @brief Handles the loading and management of image and label dataset. 
 * 
 * @note This class is non-copyable and non-movable on purpose to ensure a single ownership
 * of the data. 
 * 
 * Public Methods:
 * - DataHandler(DataSetPtr): Constructs a DataHandler with a pre-existing dataset.
 * - DataHandler(const std::string&, const std::string&): Loads data from specified file paths.
 * - Data(): Provides const access to the stored dataset.
 * - PrintShape():
 * - GetDatasetInstance():
 * 
 * 
 * Private Methods:
 * - build_data_(const std::string&, const std::string&): Static method to build the dataset from file paths.
 * - get_shape_(): Provides the shape of the data in a arrays of arrays format
 *
 */
template<typename ImageContainer, typename LabelValueType>
class DataHandler {
    public:
        using DataSetType = Data<ImageContainer, LabelValueType>; 
        using DataSetPtr = std::unique_ptr<DataSetType>;
        using DataShape = std::array<std::array<std::size_t, 2UL>, 2>; 

        using vect_size_t = std::vector<std::size_t>; 
    
        explicit DataHandler(DataSetPtr datasetptr):
            m_datasetptr(std::move(datasetptr)),
            m_shape(get_shape_()) {}

        DataHandler(const std::string& images_path, const std::string& labels_path) :
            m_datasetptr(std::move(build_data_(images_path, labels_path))),
            m_shape(get_shape_()) {}
        
        // Gets data struct underlying object non-modifiable
        const DataSetType& Data() const { 
            if (!m_datasetptr) throw std::runtime_error("Dataset pointer is null!");
            return *m_datasetptr; 
            }

        void PrintShape() const {
            std::cout << "Images shape: (" << m_shape[0][0] << ", " << m_shape[0][1] << ")\n"; 
            std::cout << "Labels shape: (" << m_shape[1][0] << ", " << m_shape[1][1] << ")\n"; 
        }

        std::pair<ImageContainer, LabelValueType> GetDatasetInstance(std::size_t index) const {
            if(index >= m_shape[0][0]){
                LOG(ERROR)  << "Out of bound value index! Index should be less than: "
                            << m_shape[0][0];
                throw std::runtime_error("Out of bounds index! Please, condider using a valid index!");
            }

            if (!m_datasetptr) throw std::runtime_error("Dataset pointer is null!");
            return {m_datasetptr->images[index],
                    m_datasetptr->labels[index]}; 
        }
    
        // void RenderImage(){}; 

        // Specifying to the compiler a vector type of images and labels iterators
        // using SplitRatio = std::tuple<double, double, double>; 

        DataHandler(const DataHandler&)= delete;
        DataHandler(DataHandler&&)= delete;
        DataHandler& operator=(const DataHandler&)= delete;
        DataHandler& operator=(DataHandler&&)= delete;
    
    private:
        DataSetPtr m_datasetptr;
        DataShape m_shape; 

        static DataSetPtr build_data_(const std::string& imgs_path, const std::string& labels_path) {

            try {

                std::vector<ImageContainer> images;
                using T = typename ImageContainer::value_type;

                LOG(INFO) << "Now reading images...";

                if constexpr(is_vector_v<ImageContainer>) {
                    images = ReadImages<T>(imgs_path);
                } else if constexpr(is_array_v<ImageContainer>){
                    constexpr std::size_t N = tuple_size_v<ImageContainer>;
                    images = ReadImages<T, N>(imgs_path);
                } else {
                    static_assert(always_false_v<ImageContainer>,
                        "Unsupported Image container! Containers handled: std::array or std::vector");
                }

                LOG(INFO) << "Images read compeleted!"; 
                LOG(INFO) << "Now reading labels..."; 

                auto labels = ReadLabels<LabelValueType>(labels_path);
                LOG(INFO) << "Labels read compeleted!";

                if(images.size() != labels.size())
                    throw std::runtime_error("Mismatch between number of images and labels");
            
                return std::make_unique<DataSetType>(DataSetType{std::move(images), std::move(labels)});
            }

            catch(const std::exception &e) {
                LOG(ERROR) << "Error in build_data_: " << e.what();
                return nullptr;
            }

        }

        DataShape get_shape_() const {
            if (!m_datasetptr) throw std::runtime_error("Dataset null pointer!");

            return DataShape {{
                {m_datasetptr->images.size(), m_datasetptr->images[0].size()},
                {m_datasetptr->labels.size(), 1UL}
            }}; 
        }

        vect_size_t data_shuffle_helper_() {

            vect_size_t shuffled_indices(m_shape[0][0]);
            std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0UL); 
            std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::mt19937{std::random_device{}()});

            return shuffled_indices;  
        }

};


}  // namespace lm
