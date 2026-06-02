#include "blocklab/Agent.h"
#include "blocklab/CudaHelpers.h"
#include "blocklab/Environment.h"
#include "blocklab/Error.h"
#include "blocklab/Observation.h"
#include "blocklab/Renderer.h"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

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
        NativeEnvironment(uint32_t numEnvs, uint32_t width, uint32_t height, uint32_t seed)
            : m_numEnvs(numEnvs)
            , m_width(width)
            , m_height(height)
            , m_rewards(std::make_unique<float[]>(numEnvs))
            , m_terminated(std::make_unique<bool[]>(numEnvs))
            , m_truncated(std::make_unique<bool[]>(numEnvs))
            , m_renderer(RenderConfig {
                  .width = static_cast<int32_t>(width),
                  .height = static_cast<int32_t>(height),
                  .batchSize = numEnvs,
                  .visible = false,
                  .present = false,
              })
            , m_environment(m_renderer, numEnvs)
        {
            if (numEnvs == 0U) [[unlikely]]
                fatalError("num_envs must be positive");
            m_renderer.setCudaObservationExportEnabled(true);
            m_environment.reset(seed);
        }

        NativeEnvironment(const NativeEnvironment&) = delete;
        NativeEnvironment& operator=(const NativeEnvironment&) = delete;

        py::dict reset(uint32_t seed)
        {
            m_environment.reset(seed);
            const Observation& observation = m_environment.observe();
            return observationInfo(observation.version());
        }

        py::dict step(const py::object& actions)
        {
            const std::vector<AgentAction> actionBatch = parseActionBatch(actions);

            const std::span<const StepResult> stepResults = m_environment.step(actionBatch);
            const Observation& observation = m_environment.observe();
            for (std::size_t i = 0; i < stepResults.size(); ++i) {
                const StepResult& stepResult = stepResults[i];
                m_rewards[i] = stepResult.reward;
                m_terminated[i] = stepResult.terminated;
                m_truncated[i] = stepResult.truncated;
            }

            py::dict result = observationInfo(observation.version());
            result["reward"] = rewardArray();
            result["terminated"] = terminatedArray();
            result["truncated"] = truncatedArray();
            return result;
        }

        Observation observe() const { return m_environment.observe(); }

        std::vector<std::size_t> lastObservationFrameIndices() const
        {
            std::vector<std::size_t> indices;
            indices.reserve(m_numEnvs);
            for (uint32_t i = 0; i < m_numEnvs; ++i)
                indices.push_back(m_renderer.lastObservationFrameIndex(i));
            return indices;
        }

        py::capsule observationDlpack(const std::vector<std::size_t>& frameIndices, uintptr_t streamHandle = 0)
        {
            if (frameIndices.size() != m_numEnvs) [[unlikely]]
                fatalError("Observation frame index count does not match environment count");
            const std::size_t frameIndex = frameIndices.front();
            for (std::size_t index : frameIndices) {
                if (index != frameIndex) [[unlikely]]
                    fatalError("Layered observations must come from the same offscreen frame");
            }

            void* const data = m_renderer.cudaObservationTensorData(frameIndex, streamHandle);
            if (!data) [[unlikely]]
                fatalError("native observation is not backed by CUDA-visible Vulkan memory");

            int deviceId = 0;
            const cudaError_t cudaResult = cudaGetDevice(&deviceId);
            if (cudaResult != cudaSuccess) [[unlikely]]
                fatalError("cudaGetDevice failed: ", cudaGetErrorString(cudaResult));

            auto* context = new DlpackObservationContext {
                .shape = { static_cast<int64_t>(m_numEnvs), 3, m_height, m_width },
                .strides = { static_cast<int64_t>(m_height) * m_width * 3, static_cast<int64_t>(m_height) * m_width,
                    m_width, 1 },
            };
            auto* managed = new DLManagedTensor {
                .dl_tensor = {
                    .data = data,
                    .device = { .device_type = DLDeviceType::Cuda, .device_id = deviceId },
                    .ndim = 4,
                    .dtype = { .code = 2, .bits = 32, .lanes = 1 },
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
        std::vector<AgentAction> parseActionBatch(const py::object& actions) const
        {
            std::vector<AgentAction> result(static_cast<std::size_t>(m_numEnvs));
            if (py::isinstance<AgentAction>(actions)) {
                std::fill(result.begin(), result.end(), py::cast<AgentAction>(actions));
                return result;
            }
            const py::sequence sequence = py::cast<py::sequence>(actions);
            const py::size_t actionCount = sequence.size();
            if (actionCount != static_cast<py::size_t>(m_numEnvs)) [[unlikely]]
                throw std::invalid_argument(
                    "actions must be an AgentAction or a sequence with num_envs AgentAction elements");
            for (py::size_t i = 0; i < actionCount; ++i)
                result[static_cast<std::size_t>(i)] = py::cast<AgentAction>(sequence[i]);
            return result;
        }

        py::dict observationInfo(uint64_t version) const
        {
            py::dict info;
            info["version"] = version;
            info["frame_indices"] = lastObservationFrameIndices();
            return info;
        }

        py::object owner() { return py::cast(this, py::return_value_policy::reference); }

        py::array_t<float> rewardArray()
        {
            return py::array_t<float>({ static_cast<py::ssize_t>(m_numEnvs) },
                { static_cast<py::ssize_t>(sizeof(float)) }, m_rewards.get(), owner());
        }

        py::array_t<bool> terminatedArray()
        {
            return py::array_t<bool>({ static_cast<py::ssize_t>(m_numEnvs) },
                { static_cast<py::ssize_t>(sizeof(bool)) }, m_terminated.get(), owner());
        }

        py::array_t<bool> truncatedArray()
        {
            return py::array_t<bool>({ static_cast<py::ssize_t>(m_numEnvs) },
                { static_cast<py::ssize_t>(sizeof(bool)) }, m_truncated.get(), owner());
        }

        std::size_t observationSliceBytes() const
        {
            return static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height) * 4U;
        }

        std::size_t observationBytes() const { return m_numEnvs * observationSliceBytes(); }

        uint32_t m_numEnvs = 0;
        uint32_t m_width = 0;
        uint32_t m_height = 0;
        std::unique_ptr<float[]> m_rewards;
        std::unique_ptr<bool[]> m_terminated;
        std::unique_ptr<bool[]> m_truncated;
        Renderer m_renderer;
        Environment m_environment;
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
        .value("RGBA8", blocklab::ObservationFormat::RGBA8)
        .value("FLOAT_NCHW", blocklab::ObservationFormat::FloatNCHW);

    py::class_<blocklab::Observation>(module, "Observation")
        .def_property_readonly("width", &blocklab::Observation::width)
        .def_property_readonly("height", &blocklab::Observation::height)
        .def_property_readonly("channels", &blocklab::Observation::channels)
        .def_property_readonly("device", &blocklab::Observation::device)
        .def_property_readonly("format", &blocklab::Observation::format)
        .def_property_readonly("version", &blocklab::Observation::version)
        .def_property_readonly("batch_size", &blocklab::Observation::batchSize)
        .def("handle", &blocklab::Observation::handle, py::arg("batch_index") = 0)
        .def("__repr__", [](const blocklab::Observation& observation) {
            return "<blocklab.Observation width=" + std::to_string(observation.width()) + " height="
                + std::to_string(observation.height()) + " channels=" + std::to_string(observation.channels())
                + " batch_size=" + std::to_string(observation.batchSize()) + " handle="
                + std::to_string(observation.handle()) + " version=" + std::to_string(observation.version()) + ">";
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
        .def_readonly("reward", &blocklab::StepResult::reward)
        .def_readonly("terminated", &blocklab::StepResult::terminated)
        .def_readonly("truncated", &blocklab::StepResult::truncated);

    py::class_<blocklab::NativeEnvironment>(module, "NativeEnvironment")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(), py::arg("num_envs") = 1,
            py::arg("width") = 160, py::arg("height") = 90, py::arg("seed") = 1)
        .def("reset", &blocklab::NativeEnvironment::reset, py::arg("seed") = 1)
        .def("step", &blocklab::NativeEnvironment::step, py::arg("actions"))
        .def("observe", &blocklab::NativeEnvironment::observe)
        .def("last_observation_frame_indices", &blocklab::NativeEnvironment::lastObservationFrameIndices)
        .def("observation_dlpack", &blocklab::NativeEnvironment::observationDlpack, py::arg("frame_indices"),
            py::arg("stream") = 0);
}
