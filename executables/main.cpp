#include <datahandler.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <filesystem>

using namespace lm;
namespace fs = std::filesystem; 

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage("General Commands");
    gflags::SetVersionString("0.0.1");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if(FLAGS_log_dir.empty()) {
        FLAGS_log_dir = ""; 
        google::SetLogDestination(google::INFO, (FLAGS_log_dir + "/INFO/").c_str());
        google::SetLogDestination(google::WARNING, (FLAGS_log_dir + "/WARNING/").c_str());
        google::SetLogDestination(google::ERROR, (FLAGS_log_dir + "/ERROR/").c_str()); 
        google::SetLogDestination(google::FATAL, (FLAGS_log_dir + "/FATAL/").c_str());
    }

    google::InitGoogleLogging(argv[0]); 

    using LabelValueType = uint8_t; 
    using ImageContainer = std::array<uint8_t, 784u>; 

    fs::path img_path = "data/train-images-idx3-ubyte";
    std::string abs_img_path = fs::absolute(img_path).string();

    fs::path lbl_path = "data/train-labels-idx1-ubyte";
    std::string abs_lbl_path = fs::absolute(lbl_path).string();

    try {
        LOG(INFO) << "Initializing Datahandler..."; 
        DataHandler<ImageContainer, LabelValueType> handler(abs_img_path, abs_lbl_path);
        LOG(INFO) << "Datahandler successfully initialized!";

        const Data<ImageContainer, LabelValueType>& data = handler.Data(); 
        handler.PrintShape();
        auto data_instance = handler.GetDatasetInstance(600);

    } catch(const std::exception& e) {
        LOG(ERROR) << "Error initializing DataHandler: " << e.what();
        return EXIT_FAILURE; 
    }
    return EXIT_SUCCESS; 
}