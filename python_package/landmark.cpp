#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nvdsmeta.h"

std::vector<std::pair<float, float>> get_landmarks_from_mask_params(NvDsObjectMeta* obj_meta) {
    std::vector<std::pair<float, float>> landmarks;

    if (obj_meta->mask_params.data != nullptr) {
        float* mask_data = obj_meta->mask_params.data;

        for (int i = 0; i < 5; i++) {
            float x = mask_data[i * 3];
            float y = mask_data[i * 3 + 1];
            landmarks.push_back(std::make_pair(x, y));
        }

        float width = obj_meta->mask_params.width;
        float height = obj_meta->mask_params.height;
        landmarks.push_back(std::make_pair(width, height));

    }

    return landmarks;
}

PYBIND11_MODULE(landmarkcus, m) {
    m.def("get_landmarks", &get_landmarks_from_mask_params, "A function to get landmarks from NvDsObjectMeta");
}