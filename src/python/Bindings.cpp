#include <blocklab/environment/Environment.h>
#include <blocklab/environment/observation/ImageBatch.h>
#include <blocklab/environment/observation/InventoryBatch.h>
#include <blocklab/environment/observation/Observation.h>
#include <blocklab/gpu/cuda/CudaHelpers.h>
#include <blocklab/gpu/vulkan/Vulkan.h>
#include <blocklab/graphics/Renderer.h>
#include <blocklab/inventory/Inventory.h>
#include <blocklab/inventory/Item.h>
#include <blocklab/utility/Error.h>

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace blocklab {
namespace {

    enum class DLDeviceType : std::int32_t {
        Cuda = 2,
    };

    struct DLDevice {
        DLDeviceType device_type;
        std::int32_t device_id;
    };

    struct DLDataType {
        std::uint8_t code;
        std::uint8_t bits;
        std::uint16_t lanes;
    };

    struct DLTensor {
        void* data;
        DLDevice device;
        std::int32_t ndim;
        DLDataType dtype;
        std::int64_t* shape;
        std::int64_t* strides;
        std::uint64_t byte_offset;
    };

    struct DLManagedTensor {
        DLTensor dl_tensor;
        void* manager_ctx;
        void (*deleter)(DLManagedTensor*);
    };

    struct DlpackObservationContext {
        std::int64_t shape[4];
        std::int64_t strides[4];
    };

    void deleteDlpackObservation(DLManagedTensor* managed)
    {
        delete static_cast<DlpackObservationContext*>(managed->manager_ctx);
        delete managed;
    }

    py::tuple itemTuple(std::span<const Item> items)
    {
        py::tuple result(items.size());
        for (std::size_t i = 0; i < items.size(); ++i)
            result[i] = py::cast(items[i]);
        return result;
    }

    class NativeEnvironment {
    public:
        NativeEnvironment(std::uint32_t numEnvs, std::uint32_t width, std::uint32_t height, std::uint32_t seed,
            std::uint32_t maxSteps)
            : m_numEnvs(numEnvs)
            , m_width(width)
            , m_height(height)
            , m_rewards(std::make_unique<float[]>(numEnvs))
            , m_terminated(std::make_unique<bool[]>(numEnvs))
            , m_truncated(std::make_unique<bool[]>(numEnvs))
            , m_vulkanInstance(false)
            , m_vulkan(m_vulkanInstance)
            , m_renderer(m_vulkan,
                  RenderConfig {
                      .width = static_cast<std::int32_t>(width),
                      .height = static_cast<std::int32_t>(height),
                      .batchSize = numEnvs,
                  })
            , m_environment(m_renderer, numEnvs, maxSteps)
        {
            if (numEnvs == 0U) [[unlikely]]
                fatalError("num_envs must be positive");
            m_environment.reset(seed);
        }

        NativeEnvironment(const NativeEnvironment&) = delete;
        NativeEnvironment& operator=(const NativeEnvironment&) = delete;

        py::dict reset(std::uint32_t seed)
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

        const Observation& observe() const { return m_environment.observe(); }

        py::capsule observationDlpack(const Observation& observation, std::uintptr_t streamHandle = 0)
        {
            const ImageBatch& images = observation.images();

            void* const data = images.data();
            if (!data) [[unlikely]]
                fatalError("native observation has no CUDA tensor data");
            if (images.batchSize() != m_numEnvs
             || images.width() != m_width
             || images.height() != m_height) [[unlikely]]
                fatalError("native observation metadata does not match environment");

            cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamHandle);
            images.enqueueReadyWait(stream);

            int deviceId = 0;
            const cudaError_t cudaResult = cudaGetDevice(&deviceId);
            if (cudaResult != cudaSuccess) [[unlikely]]
                fatalError("cudaGetDevice failed: ", cudaGetErrorString(cudaResult));

            auto* context = new DlpackObservationContext {
                .shape = { static_cast<std::int64_t>(m_numEnvs), 3, m_height, m_width },
                .strides = { static_cast<std::int64_t>(m_height) * m_width * 3,
                    static_cast<std::int64_t>(m_height) * m_width, m_width, 1 },
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

        py::dict observationInfo(std::uint64_t version) const
        {
            py::dict info;
            info["version"] = version;
            info["observation"] = py::cast(m_environment.observe(), py::return_value_policy::reference);
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

        std::uint32_t m_numEnvs = 0;
        std::uint32_t m_width = 0;
        std::uint32_t m_height = 0;
        std::unique_ptr<float[]> m_rewards;
        std::unique_ptr<bool[]> m_terminated;
        std::unique_ptr<bool[]> m_truncated;
        VulkanInstance m_vulkanInstance;
        Vulkan m_vulkan;
        Renderer m_renderer;
        Environment m_environment;
    };

} // namespace
} // namespace blocklab

PYBIND11_MODULE(_native, module)
{
    module.doc() = "Native BlockLab CUDA/Vulkan environment bindings";

    py::enum_<blocklab::Item::Type>(module, "ItemType")
        .value("Dirt", blocklab::Item::Type::Dirt)
        .value("Stone", blocklab::Item::Type::Stone)
        .value("Torch", blocklab::Item::Type::Torch);

    py::class_<blocklab::Item>(module, "Item")
        .def_property_readonly("empty", &blocklab::Item::empty)
        .def_property_readonly("type", [](const blocklab::Item& item) -> py::object {
            if (item.empty())
                return py::none();
            return py::cast(item.type());
        })
        .def_property_readonly("count", &blocklab::Item::count)
        .def("__repr__", [](const blocklab::Item& item) -> std::string {
            if (item.empty())
                return "<blocklab.Item empty>";
            return "<blocklab.Item type=" + std::to_string(static_cast<std::uint32_t>(item.type()))
                + " count=" + std::to_string(item.count()) + '>';
        });

    py::class_<blocklab::Inventory>(module, "Inventory")
        .def_property_readonly("hotbar_slots", [](const blocklab::Inventory& inventory) {
            return blocklab::itemTuple(inventory.hotbarSlots());
        })
        .def_property_readonly("storage_slots", [](const blocklab::Inventory& inventory) {
            return blocklab::itemTuple(inventory.storageSlots());
        })
        .def("__repr__", [](const blocklab::Inventory&) {
            return "<blocklab.Inventory>";
        });

    py::class_<blocklab::ImageBatch>(module, "ImageBatch")
        .def_property_readonly("width", &blocklab::ImageBatch::width)
        .def_property_readonly("height", &blocklab::ImageBatch::height)
        .def_property_readonly("channels", &blocklab::ImageBatch::channels)
        .def_property_readonly("batch_size", &blocklab::ImageBatch::batchSize)
        .def("__repr__", [](const blocklab::ImageBatch& images) {
            return "<blocklab.ImageBatch width=" + std::to_string(images.width())
                                     + " height=" + std::to_string(images.height())
                                     + " channels=" + std::to_string(images.channels())
                                     + " batch_size=" + std::to_string(images.batchSize()) + '>';
        });

    py::class_<blocklab::InventoryBatch>(module, "InventoryBatch")
        .def("__len__", &blocklab::InventoryBatch::batchSize)
        .def_property_readonly("batch_size", &blocklab::InventoryBatch::batchSize)
        .def("__getitem__", [](const blocklab::InventoryBatch& inventories, std::uint32_t index) -> const blocklab::Inventory& {
            if (index >= inventories.batchSize())
                throw py::index_error();
            return inventories[index];
        }, py::return_value_policy::reference_internal)
        .def("__repr__", [](const blocklab::InventoryBatch& inventories) {
            return "<blocklab.InventoryBatch batch_size=" + std::to_string(inventories.batchSize()) + '>';
        });

    py::class_<blocklab::Observation>(module, "Observation")
        .def_property_readonly("version", &blocklab::Observation::version)
        .def_property_readonly("batch_size", &blocklab::Observation::batchSize)
        .def_property_readonly("images", &blocklab::Observation::images, py::return_value_policy::reference_internal)
        .def_property_readonly("inventories", [](const blocklab::Observation& observation) -> const blocklab::InventoryBatch& {
            return observation.inventories();
        }, py::return_value_policy::reference_internal)
        .def("__repr__", [](const blocklab::Observation& observation) {
            return "<blocklab.Observation batch_size=" + std::to_string(observation.batchSize())
                + " version=" + std::to_string(observation.version()) + '>';
        });

    py::enum_<blocklab::PlacementBlock>(module, "PlacementBlock")
        .value("Torch", blocklab::PlacementBlock::Torch)
        .value("Dirt", blocklab::PlacementBlock::Dirt)
        .value("Stone", blocklab::PlacementBlock::Stone);

    py::class_<blocklab::AgentAction>(module, "AgentAction")
        .def(py::init<>())
        .def_readwrite("forward", &blocklab::AgentAction::forward)
        .def_readwrite("right", &blocklab::AgentAction::right)
        .def_readwrite("jump", &blocklab::AgentAction::jump)
        .def_readwrite("dig", &blocklab::AgentAction::dig)
        .def_readwrite("place", &blocklab::AgentAction::place)
        .def_readwrite("placement_block", &blocklab::AgentAction::placementBlock)
        .def_readwrite("yaw_delta", &blocklab::AgentAction::yawDelta)
        .def_readwrite("pitch_delta", &blocklab::AgentAction::pitchDelta);

    py::class_<blocklab::StepResult>(module, "StepResult")
        .def_readonly("reward", &blocklab::StepResult::reward)
        .def_readonly("terminated", &blocklab::StepResult::terminated)
        .def_readonly("truncated", &blocklab::StepResult::truncated);

    py::class_<blocklab::NativeEnvironment>(module, "NativeEnvironment")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>(),
            py::arg("num_envs") = 1, py::arg("width") = 160, py::arg("height") = 90, py::arg("seed") = 1,
            py::arg("max_steps") = 0)
        .def("reset", &blocklab::NativeEnvironment::reset, py::arg("seed") = 1)
        .def("step", &blocklab::NativeEnvironment::step, py::arg("actions"))
        .def("observe", &blocklab::NativeEnvironment::observe, py::return_value_policy::reference_internal)
        .def("observation_dlpack", &blocklab::NativeEnvironment::observationDlpack, py::arg("observation"),
            py::arg("stream") = 0);
}
