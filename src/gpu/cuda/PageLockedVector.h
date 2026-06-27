#pragma once

#include <blocklab/gpu/cuda/CudaHelpers.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

namespace blocklab {

template <typename T>
class PageLockedVector {
public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    static_assert(std::is_trivially_copyable_v<T>);

    PageLockedVector() = default;

    ~PageLockedVector()
    {
        if (m_data)
            cudaCheck(cudaFreeHost(m_data), "cudaFreeHost");
    }

    PageLockedVector(const PageLockedVector&) = delete;
    PageLockedVector& operator=(const PageLockedVector&) = delete;

    PageLockedVector(PageLockedVector&& other)
        : m_size(other.m_size)
        , m_capacity(other.m_capacity)
        , m_data(std::exchange(other.m_data, nullptr))
    {
        other.m_size = 0;
        other.m_capacity = 0;
    }

    PageLockedVector& operator=(PageLockedVector&& other)
    {
        if (this == &other)
            return *this;

        if (m_data)
            cudaCheck(cudaFreeHost(m_data), "cudaFreeHost");

        m_size = other.m_size;
        m_capacity = other.m_capacity;
        m_data = std::exchange(other.m_data, nullptr);
        other.m_size = 0;
        other.m_capacity = 0;
        return *this;
    }

    T& operator[](std::size_t i)
    {
        assert(m_data && i < m_size);
        return m_data[i];
    }

    const T& operator[](std::size_t i) const
    {
        assert(m_data && i < m_size);
        return m_data[i];
    }

    T* data() { return m_data; }
    const T* data() const { return m_data; }

    iterator begin() { return m_data; }
    const_iterator begin() const { return m_data; }
    const_iterator cbegin() const { return m_data; }

    iterator end() { return m_data + m_size; }
    const_iterator end() const { return m_data + m_size; }
    const_iterator cend() const { return m_data + m_size; }

    void reserve(std::size_t capacity)
    {
        if (capacity <= m_capacity)
            return;

        void* ptr;
        cudaCheck(cudaMallocHost(&ptr, sizeof(T) * capacity), "cudaMallocHost");
        assert(ptr);

        if (m_size) {
            assert(m_data);
            std::memcpy(ptr, m_data, m_size * sizeof(T));
        }

        if (m_data)
            cudaCheck(cudaFreeHost(m_data), "cudaFreeHost");

        m_data = static_cast<T*>(ptr);
        m_capacity = capacity;
    }

    void resize(std::size_t newSize)
    {
        reserve(newSize);
        if (newSize > m_size)
            std::fill(m_data + m_size, m_data + newSize, T {});
        m_size = newSize;
    }

    void uninitializedResize(std::size_t newSize)
    {
        reserve(newSize);
        m_size = newSize;
    }

    void push_back(T value)
    {
        if (m_capacity < m_size + 1)
            reserve(std::max(m_capacity * 2, s_initialCapacity));

        m_data[m_size++] = value;
    }

    void clear() { m_size = 0; }

    std::size_t size() const { return m_size; }

    std::size_t capacity() const { return m_capacity; }

    bool empty() const { return m_size == 0; }

private:
    static constexpr std::size_t s_initialCapacity = 4;

    std::size_t m_size = 0;
    std::size_t m_capacity = 0;
    T* m_data = nullptr;
};

} // namespace blocklab
