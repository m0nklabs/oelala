#include <openpose/headers.hpp>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    try {
        // OpenPose configuratie
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

        // Gebruik standaard pose configuratie
        op::WrapperStructPose poseWrapperStruct{};

        // Stel pose model in op BODY_25
        poseWrapperStruct.poseModel = op::PoseModel::BODY_25;

        // Configureer OpenPose
        opWrapper.configure(poseWrapperStruct);

        // Start OpenPose
        opWrapper.start();

        std::cout << "OpenPose succesvol geïnitialiseerd!" << std::endl;
        std::cout << "Pose model: BODY_25 (" << op::getPoseBodyPartMapping(op::PoseModel::BODY_25).size() << " keypoints)" << std::endl;

        // Hier kunnen we later video processing toevoegen
        std::cout << "OpenPose klaar voor gebruik in oelala project." << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
