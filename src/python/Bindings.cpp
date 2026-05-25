#include "blocklab/Agent.h"
#include "blocklab/Environment.h"
#include "blocklab/Error.h"
#include "blocklab/Observation.h"
#include "blocklab/Renderer.h"

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace blocklab {
namespace {

    enum class DLDeviceType : int32_t {
        Cuda = 2,
    };

    struct DLDevice {
        DLDeviceType device_type;
        int32_t device_id;
    };

    struct DLDataType {
        uint8_t code;
        uint8_t bits;
        uint16_t lanes;
    };

    struct DLTensor {
        void* data;
        DLDevice device;
        int32_t ndim;
        DLDataType dtype;
        int64_t* shape;
        int64_t* strides;
        uint64_t byte_offset;
    };

    struct DLManagedTensor {
        DLTensor dl_tensor;
        void* manager_ctx;
        void (*deleter)(DLManagedTensor*);
    };

    struct DlpackObservationContext {
        int64_t shape[4];
        int64_t strides[4];
    };

    void deleteDlpackObservation(DLManagedTensor* managed)
    {
        delete static_cast<DlpackObservationContext*>(managed->manager_ctx);
        delete managed;
    }

    class NativeEnvironment {
    public:
        NativeEnvironment(int32_t numEnvs, int32_t width, int32_t height, int32_t worldRadiusChunks, uint32_t seed)
            : m_environment(worldRadiusChunks)
            , m_renderer(RenderConfig {
                  .width = width,
                  .height = height,
                  .visible = false,
                  .present = false,
              })
        {
            if (numEnvs != 1)
                throw std::invalid_argument("native BlockLab backend currently supports only num_envs == 1");
            m_renderer.setCudaObservationExportEnabled(true);
            m_environment.setObservationRenderer(&m_renderer);
            m_environment.reset(seed);
        }

        Observation reset(uint32_t seed) { return m_environment.reset(seed); }

        StepResult step(const AgentAction& action) { return m_environment.step(action); }

        StepResult stepDiscrete(int32_t actionId)
        {
            AgentAction action;
            switch (actionId) {
            case 0:
                action.forward = 1.0f;
                break;
            case 1:
                action.forward = -1.0f;
                break;
            case 2:
                action.right = -1.0f;
                break;
            case 3:
                action.right = 1.0f;
                break;
            case 4:
                action.jump = true;
                break;
            case 5:
                action.dig = true;
                break;
            case 6:
                action.place = true;
                break;
            default:
                throw std::invalid_argument("unknown discrete action");
            }
            return step(action);
        }

        StepResult stepDiscreteSync(int32_t actionId)
        {
            StepResult result = stepDiscrete(actionId);
            synchronizeObservation();
            return result;
        }

        Observation observe() const { return m_environment.observe(); }

        void synchronizeObservation() { m_renderer.synchronizeObservation(); }

        py::capsule observationDlpack()
        {
            void* const data = m_renderer.cudaObservationData();
            if (!data)
                fatalError("native observation is not backed by CUDA-visible Vulkan memory");

            int deviceId = 0;
            const cudaError_t cudaResult = cudaGetDevice(&deviceId);
            if (cudaResult != cudaSuccess)
                fatalError("cudaGetDevice failed:", cudaGetErrorString(cudaResult));

            const Observation observation = m_environment.observe();
            auto* context = new DlpackObservationContext {
                .shape = { 1, observation.channels, observation.height, observation.width },
                .strides = { static_cast<int64_t>(observation.height) * observation.width * observation.channels, 1,
                    static_cast<int64_t>(observation.width) * observation.channels, observation.channels },
            };
            auto* managed = new DLManagedTensor {
                .dl_tensor = {
                    .data = data,
                    .device = { .device_type = DLDeviceType::Cuda, .device_id = deviceId },
                    .ndim = 4,
                    .dtype = { .code = 1, .bits = 8, .lanes = 1 },
                    .shape = context->shape,
                    .strides = context->strides,
                    .byte_offset = 0,
                },
                .manager_ctx = context,
                .deleter = deleteDlpackObservation,
            };
            return py::capsule(managed, "dltensor", [](PyObject* capsule) {
                if (PyCapsule_IsValid(capsule, "used_dltensor"))
                    return;
                auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
                if (tensor && tensor->deleter)
                    tensor->deleter(tensor);
            });
        }

    private:
        Environment m_environment;
        Renderer m_renderer;
    };

} // namespace
} // namespace blocklab

PYBIND11_MODULE(_native, module)
{
    module.doc() = "Native BlockLab CUDA/Vulkan environment bindings";

    py::enum_<blocklab::ObservationDevice>(module, "ObservationDevice")
        .value("NONE", blocklab::ObservationDevice::None)
        .value("CPU", blocklab::ObservationDevice::Cpu)
        .value("VULKAN_SWAPCHAIN", blocklab::ObservationDevice::VulkanSwapchain)
        .value("VULKAN_IMAGE", blocklab::ObservationDevice::VulkanImage)
        .value("CUDA", blocklab::ObservationDevice::Cuda);

    py::enum_<blocklab::ObservationFormat>(module, "ObservationFormat")
        .value("RGBA8", blocklab::ObservationFormat::RGBA8);

    py::class_<blocklab::Observation>(module, "Observation")
        .def_readonly("width", &blocklab::Observation::width)
        .def_readonly("height", &blocklab::Observation::height)
        .def_readonly("channels", &blocklab::Observation::channels)
        .def_readonly("device", &blocklab::Observation::device)
        .def_readonly("format", &blocklab::Observation::format)
        .def_readonly("handle", &blocklab::Observation::handle)
        .def_readonly("version", &blocklab::Observation::version)
        .def("__repr__", [](const blocklab::Observation& observation) {
            return "<blocklab.Observation width=" + std::to_string(observation.width)
                + " height=" + std::to_string(observation.height)
                + " channels=" + std::to_string(observation.channels)
                + " handle=" + std::to_string(observation.handle)
                + " version=" + std::to_string(observation.version) + ">";
        });

    py::class_<blocklab::AgentAction>(module, "AgentAction")
        .def(py::init<>())
        .def_readwrite("forward", &blocklab::AgentAction::forward)
        .def_readwrite("right", &blocklab::AgentAction::right)
        .def_readwrite("jump", &blocklab::AgentAction::jump)
        .def_readwrite("dig", &blocklab::AgentAction::dig)
        .def_readwrite("place", &blocklab::AgentAction::place)
        .def_readwrite("yaw_delta", &blocklab::AgentAction::yawDelta)
        .def_readwrite("pitch_delta", &blocklab::AgentAction::pitchDelta);

    py::class_<blocklab::StepResult>(module, "StepResult")
        .def_readonly("observation", &blocklab::StepResult::observation)
        .def_readonly("reward", &blocklab::StepResult::reward)
        .def_readonly("terminated", &blocklab::StepResult::terminated)
        .def_readonly("truncated", &blocklab::StepResult::truncated);

    py::class_<blocklab::NativeEnvironment>(module, "NativeEnvironment")
        .def(py::init<int32_t, int32_t, int32_t, int32_t, uint32_t>(), py::arg("num_envs") = 1,
            py::arg("width") = 160, py::arg("height") = 90, py::arg("world_radius_chunks") = 3, py::arg("seed") = 1)
        .def("reset", &blocklab::NativeEnvironment::reset, py::arg("seed") = 1)
        .def("step", &blocklab::NativeEnvironment::step, py::arg("action"))
        .def("step_discrete", &blocklab::NativeEnvironment::stepDiscrete, py::arg("action"))
        .def("step_discrete_sync", &blocklab::NativeEnvironment::stepDiscreteSync, py::arg("action"))
        .def("observe", &blocklab::NativeEnvironment::observe)
        .def("synchronize_observation", &blocklab::NativeEnvironment::synchronizeObservation)
        .def("observation_dlpack", &blocklab::NativeEnvironment::observationDlpack);
}
